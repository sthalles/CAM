# base_memory.py
import math
import torch.nn as nn
import torch
import torch.distributed as dist
from abc import ABC, abstractmethod


class BaseMemory(nn.Module, ABC):
    """
    An abstract base class for a memory module (e.g., FIFO, LFU, Hybrid).

    This class contains all the common logic for:
    - Initializing buffers (queue, age, usage, thresholds)
    - Distributed training helpers (concat_all_gather, sinkhorn_knopp)
    - Getters for queue state
    - Partitioning logic
    - Statistics reporting
    
    Subclasses must implement:
    - forward(): The main logic of the memory module.
    - update_queue(): The specific eviction policy (FIFO, LFU, etc.).
    - show_memory_healthy_report(): The policy-specific report printer.
    """

    def __init__(
        self,
        K,
        partition_size,
        out_dim=256,
        num_tasks=1,
        threshold_initial=0.5,
    ):
        # create the queue
        super().__init__()
        self.K = K
        self.partition_size = partition_size
        self.out_dim = out_dim
        self.num_tasks = num_tasks
        self.threshold_initial = threshold_initial

    def init_weights(self):
        """
        Initializes (or resets) all buffers and variables.
        
        NOTE: We use dtype=torch.float for pointers and counters (integers)
        because DINOv3's trainer attempts to fill ALL buffers with NaNs during 
        initialization checks. Int64 buffers crash when receiving NaN.
        We cast them back to long() when using them as indices.
        """
        # --- Core Memory Buffers ---
        self.register_buffer("queue", torch.randn(self.out_dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        
        # [Fix] Stored as float to prevent overflow error during train.py init
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.float))

        # [Fix] Stored as float to prevent overflow error
        self.register_buffer(
            "queue_age", torch.zeros(self.K, dtype=torch.float))

        self.register_buffer(
            "queue_usage", torch.zeros(self.K, dtype=torch.float))

        # [Fix] Stored as float
        self.register_buffer(
            "queue_last_hit_age", torch.zeros(self.K, dtype=torch.float))
        
        # [Fix] Stored as float, initialized to -1.0
        self.register_buffer("queue_label", torch.full((self.K,), -1.0, dtype=torch.long))
        
        self.register_buffer(
            "slot_thresholds",
            torch.full((self.K,), self.threshold_initial)
        )

        # --- Buffers for reporting ---
        self.register_buffer(
            "last_forgotten_mean_age_fifo", torch.tensor(0.0, dtype=torch.float))
        self.register_buffer(
            "last_forgotten_mean_usage_fifo", torch.tensor(0.0, dtype=torch.float))
        self.register_buffer(
            "last_forgotten_mean_age_priority", torch.tensor(0.0, dtype=torch.float))
        self.register_buffer(
            "last_forgotten_mean_usage_priority", torch.tensor(0.0, dtype=torch.float))

        self.register_buffer("last_actual_fifo_batch_size", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_actual_priority_batch_size", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_forgotten_mean_staleness", torch.tensor(0.0, dtype=torch.float))

        self.register_buffer("last_target_priority_batch_size", torch.tensor(1.0, dtype=torch.float))
        self.register_buffer("last_gkwta_total_hits", torch.tensor(0.0, dtype=torch.float))
        self.register_buffer("last_gkwta_potential_hits", torch.tensor(1.0, dtype=torch.float))

        self.register_buffer("last_count_refined", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_count_inserted", torch.tensor(0, dtype=torch.long))
        
        # Use uint8 for mask (safer for NCCL broadcasting than bool)
        self.register_buffer('last_batch_refinement_mask',
                             torch.zeros(1, dtype=torch.uint8), persistent=False)
        
        # --- Running Isolation Statistics ---
        self.register_buffer("running_isolation_sum", torch.tensor(0.0, dtype=torch.float))
        self.register_buffer("running_isolation_count", torch.tensor(1e-6, dtype=torch.float))
        self.register_buffer("last_reported_isolation", torch.tensor(0.0, dtype=torch.float))
        self.register_buffer("last_suppressed_count", torch.tensor(0.0, dtype=torch.float))
        
        # 2. Effective Rank (Running Average)
        self.register_buffer("running_rank_sum", torch.tensor(0.0, dtype=torch.float))
        self.register_buffer("running_rank_count", torch.tensor(1e-6, dtype=torch.float))
        
        # --- Reporting ---
        self.register_buffer("last_victim_mean_redundancy", torch.tensor(0.0))
        self.register_buffer("last_victim_mean_staleness", torch.tensor(0.0))
        self.register_buffer("last_victim_min_redundancy", torch.tensor(1.0))

    # --- Properties to safe-cast buffers for Subclasses ---
    
    @property
    def ptr_idx(self):
        """Returns queue_ptr as a long integer for indexing."""
        return self.queue_ptr.long()

    @property
    def age_idx(self):
        """Returns queue_age as long integers."""
        return self.queue_age.long()

    @property
    def label_idx(self):
        """Returns queue_label as long integers."""
        return self.queue_label.long()


    def get_features(self):
        return self.queue.clone().detach()

    def get_ages(self):
        # Cast to long for logic consistency
        return self.queue_age.long().clone().detach()

    def get_usage(self):
        return self.queue_usage.clone().detach()

    def get_last_hit_ages(self):
        return self.queue_last_hit_age.long().clone().detach()

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"K={self.K}, "
            f"partition_size={self.partition_size}, "
            f"out_dim={self.out_dim})"
        )

    def __str__(self):
        # Use self.__class__.__name__ to be generic for any subclass
        # Subclasses can override this to add more info (e.g., lfu_decay)
        return (
            f"{self.__class__.__name__}("
            f"K={self.K}, "
            f"partition_size={self.partition_size}, "
            f"out_dim={self.out_dim})"
        )
        
    @torch.no_grad()
    def compute_effective_rank(self, anchor_embeds):
        """
        Computes the Effective Rank of the current batch of anchors.
        Input: [D, P] (Features x Prototypes)
        Output: Scalar [0, 1]
        """

        # Singular values of the covariance-like structure
        singular_values = torch.linalg.svdvals(anchor_embeds)

        # 2. Normalize singular values to a probability distribution
        #    (Handling absolute zeros to avoid log(0))
        singular_values = singular_values + 1e-10
        p = singular_values / singular_values.sum()

        # 3. Shannon Entropy of the spectrum
        entropy = -torch.sum(p * torch.log(p))

        # 4. Exponentiate to get Effective Rank (Number of active dimensions)
        effective_rank = torch.exp(entropy)

        # 5. Normalize by D (Dimension) to get [0, 1] bound
        #    (Or min(P, D) if batch is smaller than Dim)
        max_possible_rank = min(anchor_embeds.shape)
        norm_rank = effective_rank / max_possible_rank

        return norm_rank.item()

    @torch.no_grad()
    def compute_usage_entropy(self):
        """
        Computes Normalized Entropy of the Usage buffer.
        Output: Scalar [0, 1]
        """
        # 1. Normalize usage to probabilities
        #    Add epsilon to avoid div-by-zero
        probs = self.queue_usage + 1e-6
        probs = probs / probs.sum()

        # 2. Shannon Entropy
        entropy = -torch.sum(probs * torch.log(probs))

        # 3. Normalize by Max Entropy (log K)
        max_entropy = math.log(self.K)
        norm_entropy = entropy / max_entropy

        return norm_entropy.item()

    @torch.no_grad()
    def measure_partition_isolation(self, anchor_embeds):
        """
        Efficient O(P^2) calculation.
        Computes isolation within the active partition (subset of memory).
        
        Args:
            anchor_embeds: Shape [D, P] (Features x Prototypes)
        """
        # 1. Transpose to [P, D] so rows are prototypes
        #    We want the Gram matrix of the prototypes (P x P)
        prototypes = anchor_embeds.T

        # 2. Compute Pairwise Similarity Matrix [P, P]
        #    (P x D) @ (D x P) -> (P x P)
        #    Since vectors are normalized, dot product = cosine similarity
        sim_matrix = torch.matmul(prototypes, prototypes.T)

        # 3. Mask Self-Similarity
        #    We want the nearest *other* neighbor.
        sim_matrix.fill_diagonal_(-1.0)

        # 4. Find Nearest Neighbor for each anchor
        #    Max along dim=1 (columns)
        max_sim, _ = sim_matrix.max(dim=1)

        # 5. Compute Isolation (1 - Similarity)
        #    Clamp to ensure non-negative (handling numerical errors)
        isolation = torch.clamp(1.0 - max_sim, min=0.0)

        # 6. Accumulate
        batch_mean = isolation.mean()
        self.running_isolation_sum += batch_mean
        self.running_isolation_count += 1.0


    @staticmethod
    @torch.no_grad()
    def concat_all_gather(tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        if not dist.is_initialized():
            return tensor
        tensors_gather = [
            torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output

    @abstractmethod
    @torch.no_grad()
    def update_queue(self, keys):
        """
        The eviction policy (e.g., FIFO, LFU, Hybrid).
        
        This method must be implemented by a subclass. It is responsible for:
        1. Gathering keys (using `concat_all_gather`).
        2. Determining which slots to evict based on the policy.
        3. Populating the relevant 'last_forgotten...' reporting buffers.
        4. Incrementing `queue_age` for all slots.
        5. Replacing evicted slots with new keys.
        6. Resetting `queue_age`, `queue_usage`, `queue_last_hit_age`, and `slot_thresholds` for new slots.
        7. Updating the `queue_ptr` if necessary (e.g., for FIFO).
        """
        raise NotImplementedError

    def get_features(self):
        return self.queue.clone().detach()

    def get_ages(self):
        return self.queue_age.clone().detach()

    def get_usage(self):
        return self.queue_usage.clone().detach()

    # NEW GETTER: For accessing last_hit_age buffer
    def get_last_hit_ages(self):
        return self.queue_last_hit_age.clone().detach()

    def get_partition_and_memory_indices(self):
        """
        Gets a random partition of indices.
        Uses the device-agnostic version from hybrid.py.
        """
        # Use the device of the module's buffers
        device = self.queue.device
        rand_cluster_indices = torch.randperm(self.K, device=device)

        partition, memory_block_indices = torch.split(
            rand_cluster_indices,
            split_size_or_sections=(
                self.partition_size, self.K - self.partition_size),
        )
        return partition, memory_block_indices

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        The main logic of the memory module.
        Must be implemented by a subclass.
        
        NOTE: Subclasses should call self.update_hits(hit_indices) after identifying
        hit slots (e.g., nearest neighbors or assignments) to increment usage and
        update last_hit_age.
        """
        raise NotImplementedError

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        # Q is K-by-B for consistency with notations from our paper
        Q = torch.exp(teacher_output / teacher_temp).t()
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    @torch.no_grad()
    def report_statistics(self):
        """
        Computes and returns a dictionary of queue health statistics.
        UPDATED: Now includes detailed Age Distribution metrics (Max, Std, Q95).
        """
        age = self.queue_age.float()
        usage = self.queue_usage.float()
        thresholds = self.slot_thresholds.float()
        last_hit_age = self.queue_last_hit_age.float()

        # Time since last hit
        last_hit_ages = age - last_hit_age

        # --- (Age, Usage, Threshold stats calculations) ---
        age_mean = age.mean().item()
        # --- NEW METRICS FOR PAPER PLOTS ---
        age_max = age.max().item()
        age_std = age.std().item()
        if age.numel() > 0:
            age_q95 = torch.quantile(age, 0.95).item()
        else:
            age_q95 = 0.0
        # -----------------------------------

        usage_mean = usage.mean().item()
        threshold_mean = thresholds.mean().item()

        # --- Compute Average Isolation from Running Buffers ---
        if self.running_isolation_count > 0.001:
            avg_isolation = self.running_isolation_sum / self.running_isolation_count
            self.last_reported_isolation.copy_(avg_isolation)

            # Reset buffers for the next reporting interval
            self.running_isolation_sum.fill_(0.0)
            self.running_isolation_count.fill_(1e-6)
        else:
            avg_isolation = self.last_reported_isolation.item()

        # Compute Approx IWU (Utility) using this running average
        # Utility = Normalized Usage * Isolation
        max_usage = max(1.0, usage.max().item())
        usage_mean_norm = usage.mean().item() / max_usage
        iwu_utility_mean = usage_mean_norm * avg_isolation

        # --- UPDATED STALENESS SCORE COMPUTATION ---
        # Formula: Staleness = TimeSinceLastHit / (log(Usage + 1) + 1)
        epsilon = 1.0
        staleness_scores = last_hit_ages / (torch.log(usage + 1) + epsilon)

        staleness_mean = staleness_scores.mean().item()
        staleness_95th_percentile = torch.quantile(
            staleness_scores, 0.95).item()
        staleness_max = staleness_scores.max().item()

        # --- (Health Metrics: Dead Slots calculation) ---
        death_threshold = 0.1

        # Changed <= 0 to < threshold
        zero_usage_slots_count = (usage < death_threshold).sum().item()
        zero_usage_pct = (zero_usage_slots_count / self.K) * 100

        if age.max().item() == 0:
            age_75th_percentile = 0.0
        else:
            age_75th_percentile = torch.quantile(age, 0.75).item()

        stale_mask = (age >= age_75th_percentile)

        # FIX: Apply threshold here too
        dead_slots_mask = stale_mask & (usage < death_threshold)
        dead_slots_count = dead_slots_mask.sum().item()
        dead_slots_total_pct = (dead_slots_count / self.K) * 100

        # --- NEW METRIC CALCULATIONS ---

        # 1. Gk-WTA Hit Rate
        total_hits = self.last_gkwta_total_hits.item()
        potential_hits = self.last_gkwta_potential_hits.item()
        gkwta_hit_rate = (total_hits / potential_hits) * \
            100 if potential_hits > 0 else 0.0

        # 2. Priority Fill Rate
        actual_priority = self.last_actual_priority_batch_size.item()
        target_priority = self.last_target_priority_batch_size.item()
        priority_fill_rate = (actual_priority / target_priority) * \
            100 if target_priority > 0 else 0.0

        # 1. Usage Fairness (0.0 to 1.0)
        usage_entropy = self.compute_usage_entropy()

        # 2. Geometry Quality (0.0 to 1.0)
        if self.running_rank_count > 0.001:
            avg_rank = self.running_rank_sum / self.running_rank_count
        else:
            avg_rank = 0.0

        stats_dict = {
            # --- General Stats ---
            "age_mean": age_mean,
            # NEW KEYS START
            "age_max": age_max,
            "age_std": age_std,
            "age_q95": age_q95,
            # NEW KEYS END
            "usage_mean": usage_mean,
            "usage_median": usage.median().item(),
            "usage_max": usage.max().item(),
            "usage_std": usage.std().item(),

            "threshold_mean": threshold_mean,
            "threshold_median": thresholds.median().item(),
            "threshold_min": thresholds.min().item(),
            "threshold_max": thresholds.max().item(),

            "staleness_mean": staleness_mean,
            "staleness_95th_percentile": staleness_95th_percentile,
            "staleness_max": staleness_max,
            "last_count_refined": self.last_count_refined.item(),
            "last_count_inserted": self.last_count_inserted.item(),

            # --- Key Health Indicators ---
            "health_zero_usage_slots_pct": zero_usage_pct,
            "health_dead_slots_total_pct": dead_slots_total_pct,

            # --- Forgotten Slot Stats ---
            "last_forgotten_mean_age_fifo": self.last_forgotten_mean_age_fifo.item(),
            "last_forgotten_mean_usage_fifo": self.last_forgotten_mean_usage_fifo.item(),
            "last_forgotten_mean_age_priority": self.last_forgotten_mean_age_priority.item(),
            "last_forgotten_mean_usage_priority": self.last_forgotten_mean_usage_priority.item(),

            # NEW METRICS
            "isolation_mean": avg_isolation,
            "iwu_utility_mean": iwu_utility_mean,

            # --- Operational Metrics ---
            "gkwta_hit_rate": gkwta_hit_rate,
            "priority_fill_rate": priority_fill_rate,
            "last_actual_fifo_batch_size": self.last_actual_fifo_batch_size.item(),
            "last_actual_priority_batch_size": self.last_actual_priority_batch_size.item(),

            "metric_usage_entropy": usage_entropy,  # [0, 1] -> 1.0 is best
            "metric_effective_rank": avg_rank,      # [0, 1] -> 1.0 is best
        }

        return stats_dict

    @abstractmethod
    @torch.no_grad()
    def show_memory_healthy_report(self):
        """
        Calls report_statistics() and prints a formatted report.
        
        This method must be implemented by a subclass, as the
        formatting may differ based on the eviction policy
        (e.g., LFU-specific notes, Hybrid dynamic stats).
        
        This should only be called from the main process (rank 0)
        to avoid duplicated print statements.
        """
        raise NotImplementedError

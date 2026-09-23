# memory.py
import math
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
# from .fifo import FIFOMemory as BaseMemory
# from .lru import LRUMemory as BaseMemory
from .density import SoftRedundancyMemory as BaseMemory
# from .lfu import LFUMemory as BaseMemory

class Memory(BaseMemory):
    def __init__(self,
                 K,
                 partition_size,
                 out_dim=256,
                 num_tasks=1,
                 threshold_initial=0.5,
                 # --- CONTROLLER PARAMS ---
                 refinement_ratio=0.1,
                 threshold_decay_rate=0.001,
                 threshold_min=0.0,
                 threshold_max=1.0,
                 # -----------------------------
                 k_winners=5,
                 target_residual=0.01,
                 refinement_momentum=0.9,
                 target_update_size=256,
                 refinement_update=True,
                 grace_period_ratio=0.25,
                 **kwargs):

        super().__init__(
            K=K,
            partition_size=partition_size,
            out_dim=out_dim,
            num_tasks=num_tasks,
            threshold_initial=threshold_initial,
            **kwargs
        )
        self.refinement_update = refinement_update
        self.k_winners = k_winners
        self.target_residual = target_residual
        self.target_update_size = target_update_size

        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.refinement_ratio = refinement_ratio
        self.refinement_momentum = refinement_momentum

        # =========================================================
        # 🧠 PHYSICS-BASED AUTOMATION
        # =========================================================
        safe_update_size = max(1, target_update_size)
        nominal_turnover = K / safe_update_size

        calculated_grace = int(nominal_turnover * grace_period_ratio)
        self.priority_grace_period = max(1, calculated_grace)

        # Automate Threshold Decay
        self.threshold_decay_rate = 1.0 / nominal_turnover

        # Automate Usage Decay
        self.usage_decay = math.pow(self.target_residual, safe_update_size / K)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"--- Memory Auto-Config (K={K}, Update={target_update_size}) ---")
            print(f"   Turnover Time : {nominal_turnover:.1f} steps")
            print(f"   Usage Decay   : {self.usage_decay:.6f}")
            print(f"   Refine Ratio  : {self.refinement_ratio}")
            print(f"   Thresh Decay  : {self.threshold_decay_rate:.6f}")

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1

    @torch.no_grad()
    def concat_all_gather(self, tensor):
        if not dist.is_initialized():
            return tensor
        tensors_gather = [torch.ones_like(tensor)
                          for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.cat(tensors_gather, dim=0)
        return output

    @torch.no_grad()
    def update_local_memory(self, global_keys, anchor_indices=None, labels=None, protected_indices=None):
        """
        Updates the memory queue. Respects the refinement mask (populated in _execute_global_update).
        """
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        # Only Rank 0 performs the update logic
        if rank == 0:
            keys_detached = global_keys.detach()
            labels_detached = labels.detach() if labels is not None else None
            N = keys_detached.shape[0]

            # Ensure mask exists
            if not hasattr(self, 'last_batch_refinement_mask') or self.last_batch_refinement_mask.shape[0] != N:
                self.last_batch_refinement_mask = torch.zeros(N, device=keys_detached.device, dtype=torch.uint8)

            mask_bool = self.last_batch_refinement_mask.bool()

            assert mask_bool.shape[0] == N, \
                f"Shape Mismatch! Mask {mask_bool.shape} vs Keys {keys_detached.shape}."

            # Filter: Only add keys that were NOT used for refinement
            keys_to_insert = keys_detached[~mask_bool]
            labels_to_insert = None
            if labels_detached is not None:
                labels_to_insert = labels_detached[~mask_bool]

            if keys_to_insert.shape[0] > 0:
                self.update_queue(keys_to_insert, explicit_indices=anchor_indices, 
                                  labels=labels_to_insert, protected_indices=protected_indices)

    @torch.no_grad()
    def update_global_memory(self, keys, anchor_indices=None, labels=None, protected_indices=None):
        """Wrapper for readability in global forward."""
        self.update_local_memory(keys, anchor_indices, labels, protected_indices)

    @torch.no_grad()
    def _execute_global_update(self, all_candidates_list, teacher_embeds, bs):
        """
        Phase 2 (Selection) & Phase 3 (Actuator).
        Selects best global matches enforcing 1-to-1 constraint and Budget.
        Executes Momentum Update.
        """
        # Reset mask for the current batch
        self.last_batch_refinement_mask = torch.zeros(bs, device=teacher_embeds.device, dtype=torch.uint8)
        
        if not all_candidates_list:
            self.last_count_refined.zero_()
            return None

        # Consolidate candidates
        # Shape: [Total_Candidates, 3] -> (batch_idx, mem_idx, sim)
        all_candidates = torch.cat(all_candidates_list, dim=0)
        
        if all_candidates.shape[0] == 0:
            self.last_count_refined.zero_()
            return None

        # --- 1. Enforce 1-to-1 Constraint (Best Match per Batch Item) ---
        # Using Scatter Reduce (amax) to find the best similarity score per batch index
        b_idxs = all_candidates[:, 0].long()
        sims = all_candidates[:, 2]
        
        # Init with -1 (sims are usually > 0, but safe bound)
        max_sims = torch.full((bs,), -1.0, device=teacher_embeds.device)
        max_sims = max_sims.scatter_reduce_(0, b_idxs, sims, reduce='amax', include_self=False)
        
        # Filter candidates: keep only those that equal the max score for their batch_idx
        # Note: In rare float tie cases, we might keep >1 candidate per batch idx, handled by slice later or negligible.
        is_best = (sims == max_sims[b_idxs])
        best_candidates = all_candidates[is_best]

        # --- 2. Global Budget Selection ---
        # Sort these best matches by similarity descending
        sort_indices = torch.argsort(best_candidates[:, 2], descending=True)
        sorted_best_candidates = best_candidates[sort_indices]

        # Apply Hard Budget
        target_budget = int(bs * self.refinement_ratio)
        final_candidates = sorted_best_candidates[:target_budget]

        if final_candidates.shape[0] == 0:
            self.last_count_refined.zero_()
            return None

        # --- 3. Execute Momentum Update ---
        final_batch_idx = final_candidates[:, 0].long()
        final_mem_idx = final_candidates[:, 1].long()

        # Update Mask so we don't re-add these to the queue later
        self.last_batch_refinement_mask[final_batch_idx] = 1

        # Momentum Update
        current_mem_vals = self.queue[:, final_mem_idx]
        teacher_vals = teacher_embeds[final_batch_idx].T

        new_mem_vals = (self.refinement_momentum * current_mem_vals) + \
                       ((1 - self.refinement_momentum) * teacher_vals)

        self.queue[:, final_mem_idx] = F.normalize(new_mem_vals, dim=0)

        # Metrics
        self.last_count_refined.copy_(torch.tensor(len(final_batch_idx), device=teacher_embeds.device))
        
        return final_mem_idx

    @torch.no_grad()
    def _collect_candidates(self, teacher_anchor_sim, anchor_indices, bs):
        """
        Phase 1 (Sensor): Identifies matches based on thresholds.
        Accumulates Usage Stats but DOES NOT update the queue values.
        Returns: 
            - candidates: Tensor [batch_idx, memory_idx, similarity] or None
            - current_hits_count: float (for reporting)
            - current_potential_hits: float (for reporting)
        """
        partition_size = teacher_anchor_sim.shape[1]
        k = min(self.k_winners, partition_size)

        # 1. Find potential winners in this partition
        topk_sims, topk_indices = torch.topk(teacher_anchor_sim, k=k, dim=1)
        
        # Map to global indices
        global_topk_indices = anchor_indices[topk_indices] # [BS, k]

        # 2. Check against Thresholds (The Truth Check)
        thresholds_for_winners = self.slot_thresholds[global_topk_indices]
        is_novel_enough_mask = topk_sims > thresholds_for_winners

        # 3. Update Usage Stats (Sensor)
        # We accumulate hits immediately as this reflects "Activation" regardless of whether we refine later.
        valid_topk_local_indices = topk_indices[is_novel_enough_mask]
        
        current_hits_count = 0.0
        current_potential_hits = float(bs * k)

        # NOTE: bincount on local indices [0..PartitionSize]
        if valid_topk_local_indices.numel() > 0:
            hits = torch.bincount(valid_topk_local_indices, minlength=partition_size)
            self.queue_usage[anchor_indices] += hits
            
            # --- Capture Stats for Reporting ---
            current_hits_count = hits.sum().item()
            
            # Update Last Hit Age
            hit_mask_local = hits > 0
            global_indices_that_hit = anchor_indices[hit_mask_local]
            if global_indices_that_hit.numel() > 0:
                self.queue_last_hit_age[global_indices_that_hit] = self.queue_age[global_indices_that_hit]

            # Update Thresholds (Homeostasis)
            if self.refinement_update:
                epsilon = self.threshold_decay_rate
                target_rho = max(self.refinement_ratio, 1e-4)
                increase_rate = epsilon * (1.0 - target_rho) / target_rho

                global_indices_no_hit = anchor_indices[~hit_mask_local]

                self.slot_thresholds[global_indices_that_hit] += increase_rate
                self.slot_thresholds[global_indices_no_hit] -= epsilon
                self.slot_thresholds[anchor_indices] = torch.clamp(
                    self.slot_thresholds[anchor_indices], self.threshold_min, self.threshold_max
                )
        else:
            # No hits in this partition, decay everyone
            if self.refinement_update:
                self.slot_thresholds[anchor_indices] = torch.clamp(
                    self.slot_thresholds[anchor_indices] - self.threshold_decay_rate, 
                    self.threshold_min, self.threshold_max
                )

        # 4. Return Candidates
        candidates = None
        if is_novel_enough_mask.any():
            batch_indices = torch.arange(bs, device=teacher_anchor_sim.device).unsqueeze(1).expand(bs, k)
            
            valid_batch_idx = batch_indices[is_novel_enough_mask]
            valid_mem_idx = global_topk_indices[is_novel_enough_mask]
            valid_sims = topk_sims[is_novel_enough_mask]
            
            # Stack into a (N, 3) tensor
            candidates = torch.stack([valid_batch_idx.float(), valid_mem_idx.float(), valid_sims], dim=1)
            
        return candidates, current_hits_count, current_potential_hits

    def forward_global(self, student_embeds, teacher_embeds, student_temp, teacher_temp, update_memory=False, true_labels=None):
        memory_embeds = self.get_features()
        student_partition_probs_list, teacher_partition_probs_list = [], []

        # 1. PRE-PROCESSING
        local_view_for_memory = teacher_embeds[1].detach()
        student_embeds_shape = student_embeds.shape
        teacher_embeds_shape = teacher_embeds.shape
        
        if len(student_embeds_shape) == 3:
            student_embeds = student_embeds.view(-1, student_embeds_shape[-1])
        if len(teacher_embeds_shape) == 3:
            teacher_embeds = teacher_embeds.view(-1, teacher_embeds_shape[-1])

        local_labels_for_memory = None
        if true_labels is not None:
            local_labels_for_memory = true_labels

        if update_memory:
            # 2. LOCAL SUBSAMPLING
            local_quota = self.target_update_size // self.world_size
            if local_view_for_memory.shape[0] > local_quota:
                indices = torch.randperm(local_view_for_memory.shape[0], device=teacher_embeds.device)[:local_quota]
                local_view_for_memory = local_view_for_memory[indices]
                if local_labels_for_memory is not None:
                    local_labels_for_memory = local_labels_for_memory[indices]

            # 3. GATHER
            if dist.is_initialized():
                global_teacher_embeds = self.concat_all_gather(local_view_for_memory)
                if local_labels_for_memory is not None:
                    global_labels = self.concat_all_gather(local_labels_for_memory)
            else:
                global_teacher_embeds = local_view_for_memory
            
            # Physics Update
            if self.K > 0:
                self.usage_decay = math.pow(self.target_residual, global_teacher_embeds.shape[0] / self.K)
        else:
            global_teacher_embeds = local_view_for_memory
            global_labels = local_labels_for_memory

        # Accumulators
        multi_task_anchor_indices = []
        all_candidate_lists = [] # For storing deferred matches
        
        # --- Stats Accumulators ---
        total_batch_hits = 0.0
        total_batch_potential = 0.0

        # 4. LOOP OVER TASKS (Logits & Candidate Search)
        for i in range(self.num_tasks):
            anchor_indices, memory_block_indices = self.get_partition_and_memory_indices()
            memory_block_indices = memory_block_indices[: 4096]

            if update_memory:
                multi_task_anchor_indices.append(anchor_indices)

            anchor_embeds = torch.take_along_dim(
                memory_embeds, indices=anchor_indices.unsqueeze(0), dim=1
            )
            candidate_support_embeds = torch.take_along_dim(
                memory_embeds, indices=memory_block_indices.unsqueeze(0), dim=1
            )

            # Logits
            student_logits = (student_embeds @ anchor_embeds)
            bs = teacher_embeds.shape[0]
            teacher_logits_local = torch.cat(
                (candidate_support_embeds.t(), teacher_embeds), dim=0) @ anchor_embeds

            teacher_probs = self.sinkhorn_knopp_teacher(
                teacher_logits_local, teacher_temp)[-bs:, :]

            if len(student_embeds_shape) == 3:
                student_logits = student_logits.view(student_embeds_shape[0], student_embeds_shape[1], -1)
            
            if len(teacher_embeds_shape) == 3:
                teacher_probs = teacher_probs.view(teacher_embeds_shape[0], teacher_embeds_shape[1], -1)

            student_partition_probs_list.append(student_logits)
            teacher_partition_probs_list.append(teacher_probs)

            # --- SENSOR PHASE (Rank 0 only) ---
            if update_memory:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    sim_to_anchors = global_teacher_embeds @ anchor_embeds
                    
                    # Collect candidates without updating queue
                    cands, hits_cnt, pot_cnt = self._collect_candidates(
                        sim_to_anchors, anchor_indices, bs=global_teacher_embeds.shape[0])
                    
                    if cands is not None:
                        all_candidate_lists.append(cands)
                    
                    # Accumulate stats from Sensor
                    total_batch_hits += hits_cnt
                    total_batch_potential += pot_cnt

        # 5. ACTUATOR PHASE (Rank 0 only, Post-Loop)
        if update_memory:
            protected_indices = None
            
            if not dist.is_initialized() or dist.get_rank() == 0:
                bs = global_teacher_embeds.shape[0]
                refined_idx_tensor = self._execute_global_update(all_candidate_lists, global_teacher_embeds, bs)
                if refined_idx_tensor is not None:
                    protected_indices = refined_idx_tensor
                
                # --- REPORTING ---
                # Write the accumulated stats to the buffers for the logger
                self.last_gkwta_total_hits.fill_(total_batch_hits)
                self.last_gkwta_potential_hits.fill_(total_batch_potential)

            # 6. QUEUE UPDATE (Respects the mask generated in _execute_global_update)
            # We use the first set of anchor indices for the density scan context if needed
            anchor_ctx = multi_task_anchor_indices[0] if len(multi_task_anchor_indices) > 0 else None
            
            self.update_global_memory(global_teacher_embeds, anchor_indices=anchor_ctx, 
                                      labels=global_labels, protected_indices=protected_indices)

            # 7. SYNC
            if dist.is_initialized():
                dist.barrier()
                if dist.get_rank() == 0:
                    # Ensure contiguous before broadcast
                    for tensor in [self.queue, self.queue_label, self.queue_ptr, self.queue_usage, 
                                   self.queue_age, self.slot_thresholds, self.queue_last_hit_age]:
                        if not tensor.is_contiguous():
                            tensor.set_(tensor.contiguous())

                dist.broadcast(self.queue, src=0)
                dist.broadcast(self.queue_label, src=0) 
                dist.broadcast(self.queue_ptr, src=0)
                dist.broadcast(self.queue_usage, src=0)
                dist.broadcast(self.queue_age, src=0)
                dist.broadcast(self.queue_last_hit_age, src=0)
                dist.broadcast(self.slot_thresholds, src=0)

        return student_partition_probs_list, teacher_partition_probs_list

    def forward_local(self, student_embeds, teacher_embeds, student_temp, teacher_temp):
        # Setup features
        memory_embeds = self.get_features()
        student_partition_probs_list, teacher_partition_probs_list = [], []
        
        # Accumulators for deferred logic
        multi_task_anchor_indices = []
        all_candidate_lists = []
        
        # --- Stats Accumulators ---
        total_batch_hits = 0.0
        total_batch_potential = 0.0

        # 1. PRE-PROCESSING & RESHAPING
        student_embeds_shape = student_embeds.shape
        teacher_embeds_shape = teacher_embeds.shape
        
        if len(student_embeds_shape) == 3:
            student_embeds = student_embeds.view(-1, student_embeds_shape[-1])
        if len(teacher_embeds_shape) == 3:
            teacher_embeds = teacher_embeds.view(-1, teacher_embeds_shape[-1])

        # 2. LOCAL SUBSAMPLING (Teacher View)
        local_quota = self.target_update_size // self.world_size
        local_teacher_view = teacher_embeds

        if local_teacher_view.shape[0] > local_quota:
            indices = torch.randperm(local_teacher_view.shape[0], device=teacher_embeds.device)[:local_quota]
            local_teacher_view = local_teacher_view[indices]

        # 3. GATHER GLOBAL TEACHER BATCH
        if dist.is_initialized():
            global_teacher_embeds = self.concat_all_gather(local_teacher_view)
        else:
            global_teacher_embeds = local_teacher_view

        # 4. PHYSICS UPDATE (Usage Decay)
        if self.K > 0 and global_teacher_embeds.shape[0] > 0:
            self.usage_decay = math.pow(self.target_residual, global_teacher_embeds.shape[0] / self.K)
        else:
            self.usage_decay = 1.0

        # 5. LOOP OVER TASKS (Logits & Candidate Sensor)
        for i in range(self.num_tasks):
            anchor_indices, memory_block_indices = self.get_partition_and_memory_indices()
            memory_block_indices = memory_block_indices[: 4096]

            multi_task_anchor_indices.append(anchor_indices)

            anchor_embeds = torch.take_along_dim(
                memory_embeds, indices=anchor_indices.unsqueeze(0), dim=1
            )
            candidate_support_embeds = torch.take_along_dim(
                memory_embeds, indices=memory_block_indices.unsqueeze(0), dim=1
            )

            # Compute Logits
            student_logits = (student_embeds @ anchor_embeds)
            bs = teacher_embeds.shape[0]
            teacher_logits_local = torch.cat(
                (candidate_support_embeds.t(), teacher_embeds), dim=0) @ anchor_embeds

            teacher_probs = self.sinkhorn_knopp_teacher(
                teacher_logits_local, teacher_temp)[-bs:, :]

            # Reshape back if necessary for output
            if len(student_embeds_shape) == 3:
                student_logits_reshaped = student_logits.view(student_embeds_shape[0], student_embeds_shape[1], -1)
            else:
                student_logits_reshaped = student_logits
            
            if len(teacher_embeds_shape) == 3:
                teacher_probs_reshaped = teacher_probs.view(teacher_embeds_shape[0], teacher_embeds_shape[1], -1)
            else:
                teacher_probs_reshaped = teacher_probs

            student_partition_probs_list.append(student_logits_reshaped)
            teacher_partition_probs_list.append(teacher_probs_reshaped)

            # --- SENSOR PHASE (Rank 0 only) ---
            if not dist.is_initialized() or dist.get_rank() == 0:
                sim_to_anchors = global_teacher_embeds @ anchor_embeds
                
                # Use the new Sensor function
                cands, hits_cnt, pot_cnt = self._collect_candidates(
                    sim_to_anchors, anchor_indices, bs=global_teacher_embeds.shape[0])
                
                if cands is not None:
                    all_candidate_lists.append(cands)
                
                # Accumulate stats
                total_batch_hits += hits_cnt
                total_batch_potential += pot_cnt

        # 6. ACTUATOR PHASE (Rank 0 only, Post-Loop)
        protected_indices = None
        if not dist.is_initialized() or dist.get_rank() == 0:
            bs = global_teacher_embeds.shape[0]
            # Use the new Actuator function
            refined_idx_tensor = self._execute_global_update(all_candidate_lists, global_teacher_embeds, bs)
            if refined_idx_tensor is not None:
                protected_indices = refined_idx_tensor

            # --- REPORTING ---
            self.last_gkwta_total_hits.fill_(total_batch_hits)
            self.last_gkwta_potential_hits.fill_(total_batch_potential)

            # 7. QUEUE UPDATE (FIFO/Replacement)
            anchor_ctx = multi_task_anchor_indices[0] if len(multi_task_anchor_indices) > 0 else None
            
            self.update_local_memory(
                global_teacher_embeds, 
                anchor_indices=anchor_ctx, 
                labels=None, 
                protected_indices=protected_indices
            )

        # 8. SYNCHRONIZATION
        if dist.is_initialized():
            dist.barrier()
            if dist.get_rank() == 0:
                for tensor in [self.queue, self.queue_label, self.queue_ptr, self.queue_usage, 
                               self.queue_age, self.slot_thresholds, self.queue_last_hit_age]:
                    if not tensor.is_contiguous():
                        tensor.set_(tensor.contiguous())

            dist.broadcast(self.queue, src=0)
            dist.broadcast(self.queue_ptr, src=0)
            dist.broadcast(self.queue_usage, src=0)
            dist.broadcast(self.queue_age, src=0)
            dist.broadcast(self.queue_last_hit_age, src=0)
            dist.broadcast(self.slot_thresholds, src=0)

        return student_partition_probs_list, teacher_partition_probs_list

    def forward(self):
        raise NotImplementedError
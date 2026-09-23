# density.py
import torch
import torch.nn as nn
import torch.distributed as dist
from abc import ABC, abstractmethod
from .base import BaseMemory

class SoftRedundancyMemory(BaseMemory, ABC):
    def __init__(
        self,
        K,
        partition_size,
        out_dim=256,
        num_tasks=1,
        threshold_initial=0.5,
        priority_ratio: float = 0.2,
        priority_grace_period: int = 10,
        dedup_threshold: float = 0.95,
        **discard
    ):
        super().__init__(
            K=K,
            partition_size=partition_size,
            out_dim=out_dim,
            num_tasks=num_tasks,
            threshold_initial=threshold_initial,
        )
        self.priority_ratio = priority_ratio
        self.priority_grace_period = priority_grace_period
        self.dedup_threshold=dedup_threshold
        
        # New Reporting buffers (Clean slate)
        self.register_buffer("last_max_sim_observed", torch.tensor(0.0))
        self.register_buffer("last_victim_sim_score", torch.tensor(0.0))

    @torch.no_grad()
    def update_queue(self, keys, explicit_indices, labels=None, protected_indices=None):
        batch_size = keys.shape[0]
        self.last_count_inserted.copy_(batch_size)
        device = keys.device
        
        # 1. Quotas (This is now a CEILING, not a floor)
        target_smart_size = int(batch_size * self.priority_ratio)
        target_fifo_size = batch_size - target_smart_size
        
        is_reserved = torch.zeros(self.K, dtype=torch.bool, device=device)

        # --- A. Base FIFO Eviction ---
        ptr = int(self.queue_ptr)
        fifo_indices = torch.arange(ptr, ptr + target_fifo_size, device=device) % self.K
        
        is_reserved[fifo_indices] = True
        if protected_indices is not None:
            is_reserved[protected_indices] = True
            
        current_ptr = (ptr + target_fifo_size) % self.K
        
        victim_list = [fifo_indices]
        actual_fifo_count = fifo_indices.numel()
        actual_smart_count = 0
        
        # --- B. Semantic Deduplication (The Razor) ---
        if target_smart_size > 0 and explicit_indices.numel() > target_smart_size:

            valid_mask = ~is_reserved[explicit_indices]
            candidates = explicit_indices[valid_mask]
            
            if candidates.numel() > target_smart_size:
                vectors = self.queue[:, candidates]
                sim_matrix = torch.matmul(vectors.T, vectors)
                sim_matrix.fill_diagonal_(-1.0)
                
                max_sim_per_item, _ = torch.max(sim_matrix, dim=1)
                
                # 1. Identify Top Candidates (The worst offenders)
                top_vals, top_indices = torch.topk(max_sim_per_item, k=target_smart_size, largest=True)
                
                # 2. APPLY THRESHOLD (The Circuit Breaker)
                # Only proceed if the similarity actually looks like a clone.
                # If dedup_threshold is 0.90, and top item is 0.65, this mask is all False.
                threshold_mask = top_vals > getattr(self, 'dedup_threshold', 0.90)
                
                real_victim_indices = top_indices[threshold_mask]
                real_victim_vals = top_vals[threshold_mask]

                # Update stats only if we actually found something
                if real_victim_vals.numel() > 0:
                    self.last_max_sim_observed.copy_(top_vals[0]) # The absolute worst one
                    self.last_victim_sim_score.copy_(real_victim_vals.mean())
                    
                    smart_victims = candidates[real_victim_indices]
                    is_reserved[smart_victims] = True
                    victim_list.append(smart_victims)
                    actual_smart_count = smart_victims.numel()
                else:
                    # Nothing crossed the threshold
                    self.last_max_sim_observed.copy_(top_vals[0]) # Report peak even if we didn't kill it
                    self.last_victim_sim_score.copy_(0.0)

        # --- C. Finalize (Auto-Balancing Fallback) ---
        final_victims = torch.cat(victim_list)
        
        # MAGIC HAPPENS HERE:
        # If 'actual_smart_count' was 0 (because of threshold), 'needed' becomes large.
        # FIFO automatically expands to fill the void.
        if final_victims.numel() < batch_size:
             needed = batch_size - final_victims.numel()
             extra_fifo = torch.arange(current_ptr, current_ptr + needed, device=device) % self.K
             final_victims = torch.cat([final_victims, extra_fifo])
             current_ptr = (current_ptr + needed) % self.K
             actual_fifo_count += needed

        self.queue_ptr[0] = current_ptr
        
        # Write Data & Metadata
        self.queue[:, final_victims] = keys.T.to(self.queue.dtype)
        if hasattr(self, 'queue_label') and labels is not None:
             self.queue_label[final_victims] = labels.to(self.queue_label.device)
             
        # Reset Stats
        self.queue_age[final_victims] = 0
        self.queue_usage[final_victims] = 0
        self.queue_last_hit_age[final_victims] = 0
        
        decay = getattr(self, 'usage_decay', 0.999) 
        self.queue_usage.mul_(decay) 
        self.queue_age += 1

        self.last_actual_fifo_batch_size.copy_(actual_fifo_count)
        self.last_actual_priority_batch_size.copy_(actual_smart_count)

    @abstractmethod
    def forward_global(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward_local(self, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def report_statistics(self):
        stats = super().report_statistics()
        
        thresholds = self.slot_thresholds.float()
        quantiles = torch.quantile(thresholds, torch.tensor([0.25, 0.5, 0.75, 0.90], device=thresholds.device))
        
        stats['thresh_q25'] = quantiles[0].item()
        stats['thresh_q50'] = quantiles[1].item()
        stats['thresh_q75'] = quantiles[2].item()
        stats['thresh_q90'] = quantiles[3].item()
        
        # FIXED: Only report what actually exists now
        stats['last_victim_sim_score'] = self.last_victim_sim_score.item()
        stats['last_max_sim_observed'] = self.last_max_sim_observed.item()
        
        return stats

    @torch.no_grad()
    def show_memory_healthy_report(self):
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        stats = self.report_statistics()
        
        # --- 1. Extract Counts ---
        refined_count = int(stats.get('last_count_refined', 0))
        smart_evicted_count = int(stats.get('last_actual_priority_batch_size', 0))
        fifo_evicted_count = int(stats.get('last_actual_fifo_batch_size', 0))
        
        total_processed = refined_count + smart_evicted_count + fifo_evicted_count
        
        # --- 2. Extract Metrics ---
        victim_sim = stats.get('last_victim_sim_score', 0.0)
        peak_sim = stats.get('last_max_sim_observed', 0.0)
        
        # Threshold Stats
        t_min = stats.get('threshold_min', 0.0)
        t_mean = stats.get('threshold_mean', 0.0)
        t_q25 = stats.get('thresh_q25', 0.0)
        t_q75 = stats.get('thresh_q75', 0.0)

        # Usage Stats
        u_mean = stats.get('usage_mean', 0.0)
        u_max = stats.get('usage_max', 0.0)

        # --- 3. Health Logic ---
        if smart_evicted_count > 0:
            if victim_sim > 0.90: 
                eviction_status = "✅ HEALTHY: Pruning Clones"
            elif victim_sim > 0.80:
                eviction_status = "⚠️  WARNING: Pruning Semi-Unique"
            else:
                eviction_status = "🚨 DANGER: Pruning Unique Items"
        else:
            eviction_status = "ℹ️  IDLE (Pure FIFO)"

        # --- 4. Render Sections ---
        flow_str = f"""
        ⚖️  Memory Budget Balance Sheet
        -----------------------------------------------------
          Refined (Momentum) : {refined_count:4d}  (Stable Anchors)
        + Semantic Evicted   : {smart_evicted_count:4d}  (Removed Clones)
        + FIFO Evicted       : {fifo_evicted_count:4d}  (Removed Stale)
        -----------------------------------------------------
        = TOTAL UPDATE SIZE  : {total_processed:4d} / {self.target_update_size} Target
        """

        eviction_str = f"""
        🗑️  Semantic Razor Status
        -----------------------------------------------------
        Status   : {eviction_status}
        Metric   : Avg Victim Sim = {victim_sim:.4f} (Peak: {peak_sim:.4f})
        Policy   : Evict if Sim > {getattr(self, 'dedup_threshold', 0.90):.2f}
        """
        
        gatekeeper_str = f"""
        🛡️  Gatekeeper Dynamics (Usage & Thresholds)
        -----------------------------------------------------
        Thresholds : {t_min:.3f} (Min) < {t_q25:.3f} (Q25) < {t_mean:.3f} (Avg) < {t_q75:.3f} (Q75)
        Usage Load : {u_mean:.2f} Avg / {u_max:.0f} Max (Decay: {getattr(self, 'usage_decay', 0.999):.4f})
        """

        temporal_str = f"""
        ⏳  Temporal Dynamics
        -----------------------------------------------------
        Age Mean : {stats['age_mean']:.1f}  (Max: {stats['age_max']:.0f})
        Age Std  : {stats['age_std']:.1f}
        """

        report_str = f"""
        =====================================================
          🧠 MEMORY HEALTH REPORT (K={self.K})
        =====================================================
        {flow_str}
        {eviction_str}
        {gatekeeper_str}
        {temporal_str}
        
        📈 Diagnostics
        -----------------------------------------------------
        Hit Rate : {stats['gkwta_hit_rate']:.2f}%
        Entropy  : {stats['metric_usage_entropy']:.4f}
        Isolation: {stats.get('isolation_mean', 0.0):.4f}
        =====================================================
        """
        print(report_str)
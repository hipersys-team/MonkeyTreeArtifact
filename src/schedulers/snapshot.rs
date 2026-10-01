use std::collections::{HashMap, VecDeque};

use crate::network::topology::Topology;
use crate::simulator::job_scheduler::JobScheduler;
use crate::simulator::ml_job::{JobId, MLJob};
use crate::system_modules::cassini::PlacementCandidate;

/// A job scheduler that uses a user-provided snapshot of placements.
///
/// For each job, the user specifies the exact host indices that each worker
/// should be placed on. The scheduler will only schedule a job when all of its
/// requested hosts are available simultaneously.
#[derive(Debug, Default)]
pub struct SnapshotScheduler {
    /// Queue of jobs waiting for their turn to be scheduled (by arrival order).
    job_queue: VecDeque<JobId>,
    /// User-provided placements: job_id -> list of host indices (length == num_workers).
    job_to_hosts: HashMap<JobId, Vec<usize>>,
}

impl SnapshotScheduler {
    /// Creates a new `SnapshotScheduler` without any placements.
    pub fn new() -> Self {
        Self { job_queue: VecDeque::new(), job_to_hosts: HashMap::new() }
    }

    /// Sets the desired placement for a specific job.
    /// The number of host indices must match the job's number of workers.
    pub fn set_job_placement(&mut self, job_id: JobId, hosts: Vec<usize>) {
        self.job_to_hosts.insert(job_id, hosts);
    }

    /// Clears a placement for a job.
    pub fn clear_job_placement(&mut self, job_id: JobId) {
        self.job_to_hosts.remove(&job_id);
    }
}

impl JobScheduler for SnapshotScheduler {
    fn try_schedule_job<T: Topology>(&mut self, job: &mut MLJob, _topology: &T, available_hosts: &[bool]) -> bool {
        // Look up placement for this job
        let desired = match self.job_to_hosts.get(&job.id) {
            Some(v) => v,
            None => panic!("No placement for job {}", job.id), // no placement yet
        };

        if desired.len() != job.num_workers {
            return false;
        }

        // Verify all requested hosts are currently available
        if desired.iter().any(|&h| h >= available_hosts.len() || available_hosts[h]) {
            return false;
        }

        // Assign workers to the specified hosts by worker_id order 0..num_workers-1
        for (worker_id, &host_index) in desired.iter().enumerate() {
            if let Some(worker) = job.workers.get_mut(&(worker_id as usize)) {
                worker.host_index = host_index;
            }
            job.worker_to_host.insert(worker_id as usize, host_index);
        }

        true
    }

    fn get_job_priority(&self, job: &MLJob) -> u64 {
        // Earlier submitted jobs get higher priority (like FIFO)
        u64::MAX - job.submit_time_us
    }

    fn notify_job_completed(&mut self, _job_id: JobId, _completion_time_ms: u64) {
        // Nothing to do; placements remain as-is unless the caller updates them.
    }

    fn get_next_job_to_schedule(&mut self) -> Option<JobId> {
        self.job_queue.front().copied()
    }

    fn enqueue_job(&mut self, job_id: JobId) {
        self.job_queue.push_back(job_id);
    }

    fn dequeue_job(&mut self) -> Option<JobId> {
        self.job_queue.pop_front()
    }
    
    fn has_queued_jobs(&self) -> bool {
        !self.job_queue.is_empty()
    }

    fn generate_placement_candidates<T: Topology>(
        &self,
        _job: &MLJob,
        _topology: &T,
        _available_hosts: &[bool],
        _max_candidates: usize,
    ) -> Vec<PlacementCandidate> {
        // Snapshot scheduler does not search placements; placements are provided externally
        Vec::new()
    }
}



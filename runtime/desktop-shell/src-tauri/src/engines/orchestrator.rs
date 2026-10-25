// SPDX-License-Identifier: LicenseRef-SPX-Proprietary
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
//
// orchestrator.rs — Rust port of unified_orchestrator.c
//
// Routes and schedules work across 3 RFT engine variants:
//   1. Optimized RFT (SIMD-accelerated)
//   2. Unitary RFT (standard quantum operations)
//   3. Vertex Quantum RFT (topological protection)

use std::sync::{Arc, Mutex, Condvar, atomic::{AtomicBool, Ordering}};
use std::collections::VecDeque;
use std::thread;

use super::types::*;
use super::rft_kernel::RftEngine;

// ── Assembly types ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum AssemblyType {
    Optimized = 0,
    Unitary = 1,
    Vertex = 2,
}

const ASSEMBLY_COUNT: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskType {
    RftTransform,
    QuantumContext,
    SemanticEncode,
    Entanglement,
}

// ── Task ─────────────────────────────────────────────────────────────────────

pub struct UnifiedTask {
    pub task_id: u32,
    pub task_type: TaskType,
    pub variant: RftVariant,
    pub input: Vec<Complex>,
    pub output: Vec<Complex>,
    pub preferred_assembly: AssemblyType,
    pub completed: bool,
}

impl UnifiedTask {
    pub fn new(task_id: u32, task_type: TaskType, variant: RftVariant, input: Vec<Complex>) -> Self {
        let len = input.len();
        Self {
            task_id,
            task_type,
            variant,
            input,
            output: vec![Complex::ZERO; len],
            preferred_assembly: AssemblyType::Optimized,
            completed: false,
        }
    }
}

// ── Assembly status ──────────────────────────────────────────────────────────

struct AssemblyStatus {
    assembly_id: AssemblyType,
    available: bool,
    busy: bool,
    queue_depth: u32,
    performance_score: f64,
}

// ── Orchestrator ─────────────────────────────────────────────────────────────

pub struct UnifiedOrchestrator {
    assemblies: Arc<Vec<Mutex<AssemblyStatus>>>,
    queue: Arc<(Mutex<VecDeque<UnifiedTask>>, Condvar)>,
    running: Arc<AtomicBool>,
    handles: Vec<thread::JoinHandle<()>>,
    results: Arc<Mutex<Vec<UnifiedTask>>>,
}

impl UnifiedOrchestrator {
    pub fn new() -> RftResult<Self> {
        let assemblies: Vec<Mutex<AssemblyStatus>> = [
            AssemblyType::Optimized,
            AssemblyType::Unitary,
            AssemblyType::Vertex,
        ]
        .iter()
        .map(|&id| {
            Mutex::new(AssemblyStatus {
                assembly_id: id,
                available: true,
                busy: false,
                queue_depth: 0,
                performance_score: 1.0,
            })
        })
        .collect();

        let assemblies = Arc::new(assemblies);
        let queue = Arc::new((Mutex::new(VecDeque::new()), Condvar::new()));
        let running = Arc::new(AtomicBool::new(true));
        let results = Arc::new(Mutex::new(Vec::new()));

        // Start scheduler thread
        let mut handles = Vec::new();
        {
            let q = Arc::clone(&queue);
            let r = Arc::clone(&running);
            let a = Arc::clone(&assemblies);
            let res = Arc::clone(&results);
            handles.push(thread::spawn(move || {
                scheduler_worker(q, r, a, res);
            }));
        }

        Ok(Self {
            assemblies,
            queue,
            running,
            handles,
            results,
        })
    }

    /// Select the optimal assembly for a given task type based on load and preferences
    pub fn select_optimal_assembly(&self, task_type: TaskType) -> AssemblyType {
        let preferences: [[AssemblyType; 3]; 4] = [
            // RftTransform
            [AssemblyType::Optimized, AssemblyType::Unitary, AssemblyType::Vertex],
            // QuantumContext
            [AssemblyType::Vertex, AssemblyType::Unitary, AssemblyType::Optimized],
            // SemanticEncode
            [AssemblyType::Optimized, AssemblyType::Vertex, AssemblyType::Unitary],
            // Entanglement
            [AssemblyType::Vertex, AssemblyType::Optimized, AssemblyType::Unitary],
        ];

        let pref_index = match task_type {
            TaskType::RftTransform => 0,
            TaskType::QuantumContext => 1,
            TaskType::SemanticEncode => 2,
            TaskType::Entanglement => 3,
        };

        let mut best = AssemblyType::Optimized;
        let mut best_score = -1.0_f64;

        for &candidate in &preferences[pref_index] {
            let idx = candidate as usize;
            if let Ok(status) = self.assemblies[idx].lock() {
                if status.available && !status.busy {
                    let score =
                        status.performance_score / (1.0 + status.queue_depth as f64);
                    if score > best_score {
                        best_score = score;
                        best = candidate;
                    }
                }
            }
        }

        best
    }

    /// Submit a task for processing
    pub fn submit_task(&self, mut task: UnifiedTask) -> RftResult<u32> {
        task.preferred_assembly = self.select_optimal_assembly(task.task_type);
        let task_id = task.task_id;

        let (lock, condvar) = &*self.queue;
        let mut queue = lock.lock().map_err(|_| RftError::Computation)?;
        queue.push_back(task);
        condvar.notify_one();

        Ok(task_id)
    }

    /// Get completed results
    pub fn get_results(&self) -> Vec<UnifiedTask> {
        let mut results = self.results.lock().unwrap();
        std::mem::take(&mut *results)
    }

    /// Shutdown the orchestrator
    pub fn shutdown(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        let (_, condvar) = &*self.queue;
        condvar.notify_all();
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

impl Drop for UnifiedOrchestrator {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// ── Scheduler worker ─────────────────────────────────────────────────────────

fn scheduler_worker(
    queue: Arc<(Mutex<VecDeque<UnifiedTask>>, Condvar)>,
    running: Arc<AtomicBool>,
    assemblies: Arc<Vec<Mutex<AssemblyStatus>>>,
    results: Arc<Mutex<Vec<UnifiedTask>>>,
) {
    let (lock, condvar) = &*queue;
    while running.load(Ordering::SeqCst) {
        let mut q = lock.lock().unwrap();
        while q.is_empty() && running.load(Ordering::SeqCst) {
            q = condvar.wait(q).unwrap();
        }
        if !running.load(Ordering::SeqCst) {
            break;
        }
        if let Some(mut task) = q.pop_front() {
            drop(q); // Release lock before processing

            let assembly = task.preferred_assembly;
            let idx = assembly as usize;

            // Mark assembly as busy
            if let Ok(mut status) = assemblies[idx].lock() {
                status.busy = true;
                status.queue_depth += 1;
            }

            // Route task to the appropriate RFT processor
            let result = route_task_to_assembly(&mut task, assembly);
            task.completed = result.is_ok();

            // Mark assembly as free
            if let Ok(mut status) = assemblies[idx].lock() {
                status.busy = false;
                status.queue_depth = status.queue_depth.saturating_sub(1);
            }

            // Store result
            results.lock().unwrap().push(task);
        }
    }
}

// ── Assembly processor routing ───────────────────────────────────────────────

fn route_task_to_assembly(task: &mut UnifiedTask, assembly: AssemblyType) -> RftResult<()> {
    let n = task.input.len();
    if n == 0 {
        return Err(RftError::InvalidParam);
    }

    let flags = match assembly {
        AssemblyType::Optimized => RftFlags::OPTIMIZE_SIMD,
        AssemblyType::Unitary => RftFlags::DEFAULT,
        AssemblyType::Vertex => RftFlags::TOPOLOGICAL,
    };

    let mut engine = RftEngine::new(n, flags)?;

    if task.variant != RftVariant::Standard {
        engine.set_variant(task.variant, true)?;
    }

    task.output = engine.forward(&task.input)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orchestrator_basic() {
        let mut orch = UnifiedOrchestrator::new().unwrap();
        let input = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(-1.0, 0.0),
            Complex::new(0.0, -1.0),
        ];
        let task = UnifiedTask::new(1, TaskType::RftTransform, RftVariant::Standard, input);
        orch.submit_task(task).unwrap();
        // Give scheduler time to process
        std::thread::sleep(std::time::Duration::from_millis(100));
        let results = orch.get_results();
        assert!(!results.is_empty());
        assert!(results[0].completed);
        orch.shutdown();
    }
}

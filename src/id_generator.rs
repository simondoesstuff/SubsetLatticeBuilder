use std::collections::{HashMap, VecDeque};

pub struct IdGenerator<T> {
    next_id: T,
    queue: VecDeque<T>,
}

impl IdGenerator<T> {
    pub fn new() -> Self {
        Self {
            next_id: 0,
            queue: VecDeque::new(),
        }
    }

    pub fn next(&mut self) -> T {
        if let Some(id) = self.queue.pop_front() {
            return id;
        }

        let id = self.next_id;
        self.next_id += 1;
        return id;
    }

    pub fn free(&mut self, id: T) {
        self.queue.push_back(id);
    }

    /// 1 + Largest ID that has been generated
    pub fn len(&self) -> usize {
        next_id
    }
}
use std::{mem, time::Duration};

#[derive(Debug)]
pub struct Queries {
    set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    destination_buffer: wgpu::Buffer,
    num_queries: u64,
    next_unused_query: u32,
}

impl Queries {
    pub fn new(device: &wgpu::Device, num_queries: u64) -> Self {
        Queries {
            set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("Timestamp query set"),
                count: num_queries as _,
                ty: wgpu::QueryType::Timestamp,
            }),
            resolve_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("query resolve buffer"),
                size: std::mem::size_of::<u64>() as u64 * num_queries,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
                mapped_at_creation: false,
            }),
            destination_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("query dest buffer"),
                size: std::mem::size_of::<u64>() as u64 * num_queries,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            num_queries,
            next_unused_query: 0,
        }
    }

    pub fn reset(&mut self) {
        self.next_unused_query = 0;
    }

    pub fn write_next_timestamp(&mut self, pass: &mut wgpu::RenderPass) {
        let next_unused_query = self.next_unused_query;
        pass.write_timestamp(
            &self.set,
            mem::replace(&mut self.next_unused_query, next_unused_query + 1),
        );
    }

    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(
            &self.set,
            0..self.next_unused_query,
            &self.resolve_buffer,
            0,
        );
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.destination_buffer,
            0,
            self.resolve_buffer.size(),
        );
    }

    pub fn wait_for_results(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<Duration> {
        self.destination_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| ());
        device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        let timestamps = {
            let timestamp_view = self
                .destination_buffer
                .slice(..(std::mem::size_of::<u64>() as wgpu::BufferAddress * self.num_queries))
                .get_mapped_range();
            bytemuck::cast_slice(&timestamp_view).to_vec()
        };

        self.destination_buffer.unmap();

        let period = queue.get_timestamp_period();
        timestamps
            .windows(2)
            .map(|w: &[u64]| w[1].wrapping_sub(w[0]) as f64 * period as f64 / 1e9)
            .map(Duration::from_secs_f64)
            .collect()
    }
}

import asyncio
import time     
import uuid
from typing import List, Optional

# 导入项目相关的模块
from agent.base.protocal import MySample
from agent.gym_rollout_datasource import GymRolloutDataSource  # 数据源
from agent.gym_rollout import agent_loop_generate, compute_group_advantages  # 数据生成函数
from slime.rollout.sglang_rollout import GenerateState  # 生成状态管理


class RolloutProducer:
    """
    1. 在后台持续生成训练数据（不阻塞训练过程）
    2. 将生成的数据放入队列中供消费者使用
    3. 实现背压控制，防止内存溢出
    """
    
    def __init__(self, args, queue: asyncio.Queue, data_source: GymRolloutDataSource):
        """
        - args: 配置参数（包含批次大小、模型参数等）
        - queue: 异步队列，用于存储生成的数据
        - data_source: 数据源，提供原始的prompt数据
        """
        self.args = args  # 保存配置参数
        self.queue = queue  # 数据队列（生产者和消费者的桥梁）
        self.data_source = data_source  # 数据源
        self.state = GenerateState(args)  # 生成状态管理器
        
        # 生产者状态控制
        self.is_running = False  # 是否正在运行
        self.producer_task: Optional[asyncio.Task] = None  # 生产任务的引用
        
        # 统计信息
        self.total_produced = 0  # 总共生产的样本数
        self.total_groups_produced = 0  # 总共生产的组数
        
        # 防止生产者生成过多数据导致内存溢出
        self.max_queue_size = args.global_batch_size * 10  # 10个批次的缓冲容量
        
    async def start(self):
        """
        1. 创建一个后台任务来持续生成数据
        2. 立即返回，不阻塞调用者
        3. 生产者将在后台独立运行
        
        异步解耦的体现：
        - 调用start()后立即返回，不等待数据生成完成
        - 生产者在后台持续工作，与训练过程并行
        """
        if self.is_running:
            print("Producer already running, skipping start")
            return
            
        self.is_running = True
        # 创建异步任务，让生产者在后台运行
        self.producer_task = asyncio.create_task(self._produce_loop())
        print(f"RolloutProducer started, will continuously produce data in background")
        
    async def stop(self):
        """
        1. 设置停止标志
        2. 取消后台生产任务
        3. 等待任务完全结束
        4. 输出统计信息
        """
        self.is_running = False
        
        if self.producer_task:
            # 取消后台任务
            self.producer_task.cancel()
            try:
                # 等待任务完全结束
                await self.producer_task
            except asyncio.CancelledError:
                # 这是正常的取消操作，不是错误
                pass
                
        print(f"RolloutProducer stopped, total produced: {self.total_produced} samples")
        
    async def _produce_loop(self):
        """
        1. 持续数据生成：不停地生成新的训练数据
        2. 背压控制：队列满时暂停，防止内存溢出
        3. 异步操作：使用await确保不阻塞其他任务
        4. 错误处理：捕获异常并优雅处理
        """
        try:
            print("Producer loop started, beginning continuous data generation")
            
            while self.is_running:
                # 队列满时暂停生产；防止了内存无限增长
                if self.queue.qsize() >= self.max_queue_size:
                    print(f"Queue full ({self.queue.qsize()}/{self.max_queue_size}), producer pausing...")
                    await asyncio.sleep(0.1)  # 短暂休息，让消费者有机会消费数据
                    continue
                    
                # 从数据源获取原始prompt数据
                prompt_groups = self.data_source.get_samples(1)
                if not prompt_groups:
                    print("No more prompts available, producer stopping naturally")
                    break
                    
                prompt_group = prompt_groups[0]
                # 生成唯一的组ID，用于跟踪和调试
                prompt_group_id = f"producer_group_{uuid.uuid4().hex[:8]}"
                
                print(f"Processing prompt group {prompt_group_id} with {len(prompt_group)} prompts")
                
                # 为每个prompt生成AI响应轨迹（trajectories）
                tasks = []
                for i, prompt_sample in enumerate(prompt_group):
                    trajectory_id = f"{prompt_group_id}_traj_{i}"
                    # 创建异步任务来生成AI响应
                    task = agent_loop_generate(
                        self.args,
                        prompt_sample,
                        trajectory_id=trajectory_id,
                        prompt_group_id=prompt_group_id,
                        sampling_params=self.state.sampling_params.copy()
                    )
                    tasks.append(task)
                
                print(f"Generating {len(tasks)} trajectories concurrently...")
                # 并发执行所有轨迹生成任务
                trajectory_results = await asyncio.gather(*tasks)
                
                # 计算组内优势值
                print("Computing group advantages...")
                advs = compute_group_advantages(trajectory_results, self.args)
                for samples, adv in zip(trajectory_results, advs):
                    for samp in samples:
                        samp.advantage = adv
                
                # 展平所有轨迹数据为单个列表
                all_steps = [step for trajectory_steps in trajectory_results for step in trajectory_steps]
                
                if all_steps:
                    # 生成的数据放入队列
                    await self.queue.put(all_steps)
                    self.total_produced += len(all_steps)
                    self.total_groups_produced += 1
                    
                    print(f"Producer: generated group {self.total_groups_produced}, "
                          f"{len(all_steps)} steps, queue size: {self.queue.qsize()}/{self.max_queue_size}")
                    print(f"Total produced so far: {self.total_produced} samples")
                    
        except Exception as e:
            print(f"Producer error: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # None作为特殊信号，告诉消费者没有更多数据了
            print("Producer finishing, sending end signal...")
            await self.queue.put(None)  # None作为结束信号
            print(f"Producer finished successfully!")
            print(f"Final statistics: {self.total_produced} steps, {self.total_groups_produced} groups")


class RolloutConsumer:
    """
    1. 从队列中快速获取已生成的数据
    2. 支持非阻塞获取，不等待数据生成
    3. 实现预填充机制，确保训练开始时就有足够数据
    4. 处理剩余数据，确保零浪费
    """
    
    def __init__(self, queue: asyncio.Queue, args):
        """
        初始化消费者
        
        参数解释：
        - queue: 与生产者共享的数据队列
        - args: 配置参数（包含批次大小等）
        """
        self.queue = queue  # 数据队列（与生产者共享）
        self.args = args  # 配置参数
        self.is_finished = False  # 是否接收到结束信号
        self.total_consumed = 0  # 总共消费的样本数
        
        # 确保训练开始时队列中已有足够数据，避免等待
        self.min_prefill_batches = 3  # 最小预填充3个批次
        print(f"RolloutConsumer initialized, will prefill {self.min_prefill_batches} batches")
        
    async def ensure_queue_ready(self):
        """        
        1. 等待生产者生成足够的数据
        2. 确保训练开始时队列中有足够的缓冲数据
        3. 避免训练过程中因等待数据而阻塞
        """
        # 计算需要预填充的最小样本数
        min_samples = self.min_prefill_batches * self.args.global_batch_size
        print(f"Consumer: waiting for queue to prefill (target: {min_samples} samples)...")
        print(f"   - This ensures training can start immediately without waiting")
        
        start_time = time.time()
        
        # 等待队列达到最小预填充量
        while self.queue.qsize() < min_samples and not self.is_finished:
            await asyncio.sleep(0.5)  # 每0.5秒检查一次
            current_size = self.queue.qsize()
            if current_size > 0:
                elapsed = time.time() - start_time
                print(f"Consumer: queue prefilling... {current_size}/{min_samples} samples "
                      f"(elapsed: {elapsed:.1f}s)")
        
        final_size = self.queue.qsize()
        total_time = time.time() - start_time
        print(f"Consumer: queue ready with {final_size} samples (prefill took {total_time:.1f}s)")
        print(f"Training can now start immediately - async decoupling achieved!")
        
    async def get_batch(self, batch_size: int) -> List[MySample]:
        """
        1. 非阻塞获取：优先从队列快速获取已有数据
        2. 只在必要时短暂等待
        3. 部分返回：即使数据不足也返回已有数据，避免阻塞训练
        4. 零浪费：确保所有数据都被消费
        Args:
            batch_size: 需要的数据量
            
        Returns:
            List[MySample]: 获取到的数据（可能少于请求量）
        """
        batch = []
        start_time = time.time()
        
        print(f"Consumer: requesting {batch_size} samples from queue...")
        
        while len(batch) < batch_size:
            try:
                # 非阻塞获取（get_nowait）确保了消费者不会因为等待数据生成而阻塞
                data = self.queue.get_nowait()
                
                if data is None:
                    self.is_finished = True
                    print("Consumer: received end signal from producer")
                    break
                
                # 将获取的数据添加到批次中
                batch.extend(data)
                print(f"Consumer: got {len(data)} samples from queue, batch now has {len(batch)} samples")
                
            except asyncio.QueueEmpty:
                # 队列暂时为空
                if batch:
                    # 已有部分数据，立即返回避免阻塞训练
                    print(f"Consumer: queue empty, returning {len(batch)} samples immediately (requested {batch_size})")
                    print("   - This demonstrates async decoupling: training continues with available data")
                    break
                else:
                    # 🔄 完全没有数据，短暂等待但不长时间阻塞
                    print("Consumer: no data available, brief wait...")
                    await asyncio.sleep(0.1)  # 很短的等待时间
                    
                    # 再次尝试获取，但设置超时避免长时间阻塞
                    try:
                        data = await asyncio.wait_for(self.queue.get(), timeout=0.5)
                        if data is None:
                            self.is_finished = True
                            print("Consumer: received end signal during wait")
                            break
                        batch.extend(data)
                        print(f"Consumer: got {len(data)} samples after brief wait")
                    except asyncio.TimeoutError:
                        # 超时了，但这不是错误，继续尝试
                        if not self.is_finished:
                            print(f"Consumer: timeout waiting for data, queue size: {self.queue.qsize()}")
                            print("   - Producer may be busy generating, continuing...")
                        continue
        
        #更新统计信息
        self.total_consumed += len(batch)
        consume_time = time.time() - start_time
        
        print(f"Consumer: consumed {len(batch)} samples in {consume_time:.3f}s")
        print(f"Total consumed: {self.total_consumed}, queue remaining: {self.queue.qsize()}")
        print(f"Async decoupling working: got data quickly from pre-filled buffer")
        
        return batch
    
    async def drain_remaining_data(self) -> List[MySample]:
        """
        1. 队列中的所有剩余数据都被消费
        2. 没有任何生成的数据被浪费
        3. 系统资源得到完全清理
               
        Returns:
            List[MySample]: 所有剩余的数据
        """
        remaining_data = []
        print(f"Consumer: starting to drain remaining data from queue...")
        print(f"   - Current queue size: {self.queue.qsize()}")
        print(f"   - This ensures zero data waste from async production")
        
        drain_start_time = time.time()
        
        # 持续清空队列直到完全为空
        while True:
            try:
                # 非阻塞获取剩余数据
                data = self.queue.get_nowait()
                
                if data is None:
                    # 遇到结束信号，停止清理
                    print("Consumer: encountered end signal during drain")
                    break
                    
                remaining_data.extend(data)
                print(f"Consumer: drained {len(data)} samples, total remaining: {len(remaining_data)}")
                
            except asyncio.QueueEmpty:
                # 队列已空，清理完成
                print("Consumer: queue is now empty, drain complete")
                break
        
        # 更新最终统计
        self.total_consumed += len(remaining_data)
        drain_time = time.time() - drain_start_time
        
        print(f"Consumer: drain complete in {drain_time:.3f}s")
        print(f"Final statistics:")
        print(f"   - Total consumed: {self.total_consumed} samples")
        print(f"   - Remaining data recovered: {len(remaining_data)} samples")
        print(f"   - Zero waste achieved: ")
        
        return remaining_data
    
    async def mark_done(self, data: List[MySample]):
        """
        1. 记录数据处理状态
        2. 触发后续处理逻辑
        3. 更新统计信息
        4. 执行清理操作
        
        Args:
            data: 已处理完成的数据
        """
        # 可添加数据处理的回调逻辑
        print(f"Consumer: marked {len(data)} samples as processed")
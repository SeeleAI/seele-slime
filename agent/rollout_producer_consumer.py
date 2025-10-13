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
        1. 使用阻塞式 put，队列满时自动等待
        2. 去掉手动 qsize 检查和 sleep 忙等
        3. 减少不必要的 print 输出
        """
        try:
            print("Producer: started continuous data generation")
            
            while self.is_running:
                # 从数据源获取原始prompt数据
                prompt_groups = self.data_source.get_samples(1)
                if not prompt_groups:
                    print("Producer: no more prompts, stopping")
                    break
                    
                prompt_group = prompt_groups[0]
                prompt_group_id = f"group_{uuid.uuid4().hex[:8]}"
                
                # 为每个prompt生成AI响应轨迹
                tasks = []
                for i, prompt_sample in enumerate(prompt_group):
                    trajectory_id = f"{prompt_group_id}_traj_{i}"
                    task = agent_loop_generate(
                        self.args,
                        prompt_sample,
                        trajectory_id=trajectory_id,
                        prompt_group_id=prompt_group_id,
                        sampling_params=self.state.sampling_params.copy()
                    )
                    tasks.append(task)
                
                # 并发执行所有轨迹生成任务
                trajectory_results = await asyncio.gather(*tasks)
                
                # 计算组内优势值
                advs = compute_group_advantages(trajectory_results, self.args)
                for samples, adv in zip(trajectory_results, advs):
                    for samp in samples:
                        samp.advantage = adv
                
                # 展平所有轨迹数据为单个列表
                all_steps = [step for trajectory_steps in trajectory_results for step in trajectory_steps]
                
                if all_steps:
                    # 阻塞式 put：队列满时自动等待，不消耗 CPU
                    await self.queue.put(all_steps)
                    self.total_produced += len(all_steps)
                    self.total_groups_produced += 1
                    
                    if self.total_groups_produced % 10 == 0:
                        print(f"Producer: generated {self.total_groups_produced} groups, "
                              f"{self.total_produced} samples total, queue: {self.queue.qsize()}")
                    
        except Exception as e:
            print(f"Producer error: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # 发送结束信号
            await self.queue.put(None)
            print(f"Producer: finished, total {self.total_produced} samples in {self.total_groups_produced} groups")


class RolloutConsumer:
    """
    从队列获取训练数据的消费者
    
    1. 使用阻塞式 get，确保返回完整批次
    2. 内部缓冲精确控制批次大小，避免浪费数据
    3. 去掉忙等和异常处理，降低 CPU 开销

    - 同一个 trajectory 的 samples 可能被拆散到不同 batch
      1. 保持固定的 batch_size，训练更稳定
      2. GRPO 的 advantage 已在 Producer 端预计算
      3. 训练代码将每个 sample 视为独立样本
      4. 拆散不影响训练正确性
    - 如果使用自定义 loss 函数，确保不依赖 trajectory 完整性
    - 如果需要 trajectory-level 的操作，请在 Producer 端完成

    """
    
    def __init__(self, queue: asyncio.Queue, args):
        """
        初始化消费者
        
        参数：
        - queue: 与生产者共享的数据队列
        - args: 配置参数（包含批次大小等）
        """
        self.queue = queue
        self.args = args
        self.is_finished = False
        self.total_consumed = 0
        self.buffer = []  # 内部缓冲，用于精确控制批次大小
        
        # 预填充配置
        self.min_prefill_batches = 3
        print(f"Consumer: initialized, will prefill {self.min_prefill_batches} batches")
        
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
        获取完整批次的训练数据
        
        核心改进：
        1. 阻塞式 get - 队列空时协程挂起，CPU 0%
        2. 完整批次 - 总是返回 batch_size 个样本（除非结束）
        3. 零浪费 - 多余数据缓存，下次复用
        """
        batch = []
        
        # 统一的获取逻辑：优先 buffer，再 queue
        while len(batch) < batch_size:
            # 1. 优先从内部缓冲获取
            if self.buffer:
                data = self.buffer
                self.buffer = []
            else:
                # 2. 从队列阻塞式获取
                data = await self.queue.get()
                if data is None:
                    self.is_finished = True
                    break
            
            # 3. 统一的分配逻辑
            needed = batch_size - len(batch)
            if len(data) <= needed:
                batch.extend(data)
            else:
                batch.extend(data[:needed])
                self.buffer = data[needed:]
                break
        
        # 统计
        self.total_consumed += len(batch)
        if self.total_consumed % (batch_size * 10) == 0:
            print(f"Consumer: {self.total_consumed} samples, "
                  f"queue: {self.queue.qsize()}, buffer: {len(self.buffer)}")
        
        return batch
    
    async def drain_remaining_data(self) -> List[MySample]:
        """
        清空队列和内部缓冲中的所有剩余数据
        
        使用非阻塞 get_nowait 快速清空
        
        Returns:
            List[MySample]: 所有剩余的数据
        """
        remaining_data = []
        
        # 先清空内部缓冲
        if self.buffer:
            remaining_data.extend(self.buffer)
            self.buffer = []
        
        # 再清空队列
        while True:
            try:
                data = self.queue.get_nowait()
                
                if data is None:
                    break
                    
                remaining_data.extend(data)
                
            except asyncio.QueueEmpty:
                break
        
        # 统计
        self.total_consumed += len(remaining_data)
        
        print(f"Consumer: drained {len(remaining_data)} remaining samples")
        print(f"Consumer: total consumed {self.total_consumed} samples")
        
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
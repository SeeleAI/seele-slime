import json
import random
from typing import Dict, Any, List

class DatasetGenerator:
    def __init__(self):
        # 操作符映射
        self.operators = {
            "☆": "star_operation",    # a + b - 1
            "❀": "flower_operation",  # a * 2 + b
            "☽": "moon_operation",    # (a + b) * 2
            "☀": "sun_operation"      # a * b - a
        }
        
    def _execute_tool(self, function_name: str, arguments: Dict[str, Any]) -> int:
        """执行工具函数"""
        a = int(arguments["a"])
        b = int(arguments["b"])
        
        if function_name == "star_operation":
            return a + b - 1
        elif function_name == "flower_operation":
            return a * 2 + b
        elif function_name == "moon_operation":
            return (a + b) * 2
        elif function_name == "sun_operation":
            return a * b - a
        else:
            raise ValueError(f"Unknown function: {function_name}")
    
    def evaluate_expression(self, expression: str) -> int:
        """计算表达式的值（从左到右）"""
        tokens = expression.split()
        
        if len(tokens) < 3:
            raise ValueError("Invalid expression")
        
        # 第一个数字
        result = int(tokens[0])
        
        # 从左到右处理操作符和数字
        i = 1
        while i < len(tokens) - 1:
            operator = tokens[i]
            operand = int(tokens[i + 1])
            
            function_name = self.operators[operator]
            result = self._execute_tool(function_name, {"a": result, "b": operand})
            i += 2
        
        return result
    
    def is_valid_for_elementary(self, expression: str) -> bool:
        """检查表达式是否适合小学生（结果为正整数且不太大）"""
        try:
            result = self.evaluate_expression(expression)
            # 确保结果是正整数且不超过100（适合小学生）
            return isinstance(result, int) and 1 <= result <= 100
        except:
            return False
    
    def generate_expression(self, num_operators: int, max_attempts: int = 1000) -> tuple:
        """生成适合小学生的随机表达式"""
        for _ in range(max_attempts):
            # 使用更小的数值范围
            expression_parts = [str(random.randint(1, 6))]  # 第一个数字1-6
            
            # 生成操作符和数字对
            for i in range(num_operators):
                operator = random.choice(list(self.operators.keys()))
                
                # 根据操作符类型选择合适的数字范围
                if operator == "☆":  # a + b - 1，需要确保 a + b > 1
                    number = random.randint(1, 5)
                elif operator == "❀":  # a * 2 + b，结果会比较大
                    number = random.randint(1, 4)
                elif operator == "☽":  # (a + b) * 2，结果会很大
                    number = random.randint(1, 3)
                elif operator == "☀":  # a * b - a，需要确保 a * b > a
                    number = random.randint(2, 5)
                else:
                    number = random.randint(1, 5)
                
                expression_parts.extend([operator, str(number)])
            
            expression = " ".join(expression_parts)
            
            # 检查是否适合小学生
            if self.is_valid_for_elementary(expression):
                answer = self.evaluate_expression(expression)
                return expression, answer
        
        # 如果尝试多次都没有生成合适的表达式，返回一个简单的
        simple_expression = f"{random.randint(2, 5)} ☆ {random.randint(2, 4)}"
        answer = self.evaluate_expression(simple_expression)
        return simple_expression, answer
    
    def generate_dataset(self, 
                        train_size: int = 8000, 
                        test_size: int = 2000,
                        min_operators: int = 1,
                        max_operators: int = 4) -> tuple:
        """生成训练和测试数据集"""
        
        def create_sample(expression: str, answer: int, split: str, index: int) -> dict:
            question = f"Calculate the following expression: {expression}"
            prompt_content = f"{question} Let's think step by step and output the final answer after \"####\"."
            
            return {
                "prompt": [{"role": "user", "content": prompt_content}],
                "label": str(answer),
                "metadata": {
                    "split": split,
                    "index": index,
                    "answer": answer,
                    "question": question,
                    "expression": expression,
                    "num_operators": expression.count("☆") + expression.count("❀") + expression.count("☽") + expression.count("☀"),
                    "env_name": "calc"
                }
            }
        
        train_data = []
        test_data = []
        
        # 创建操作符数量的选择列表和对应权重
        operator_choices = list(range(min_operators, max_operators + 1))
        
        # 根据操作符数量调整权重
        if len(operator_choices) == 1:
            weights = [1.0]
        elif len(operator_choices) == 2:
            weights = [0.6, 0.4]
        elif len(operator_choices) == 3:
            weights = [0.5, 0.3, 0.2]
        elif len(operator_choices) == 4:
            weights = [0.4, 0.3, 0.2, 0.1]
        else:
            # 如果有更多选择，平均分配权重
            weights = [1.0 / len(operator_choices)] * len(operator_choices)
        
        print("正在生成训练数据...")
        # 生成训练数据
        for i in range(train_size):
            if i % 1000 == 0:
                print(f"已生成训练数据: {i}/{train_size}")
            
            num_ops = random.choices(operator_choices, weights=weights)[0]
            expression, answer = self.generate_expression(num_ops)
            sample = create_sample(expression, answer, "train", i)
            train_data.append(sample)
        
        print("正在生成测试数据...")
        # 生成测试数据
        for i in range(test_size):
            if i % 500 == 0:
                print(f"已生成测试数据: {i}/{test_size}")
            
            num_ops = random.choices(operator_choices, weights=weights)[0]
            expression, answer = self.generate_expression(num_ops)
            sample = create_sample(expression, answer, "test", train_size + i)
            test_data.append(sample)
        
        return train_data, test_data
    
    def save_to_jsonl(self, data: List[dict], filename: str):
        """保存数据到JSONL文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def analyze_dataset(self, data: List[dict]):
        """分析数据集的统计信息"""
        answers = [int(item['label']) for item in data]
        num_operators = [item['metadata']['num_operators'] for item in data]
        
        print(f"答案范围: {min(answers)} - {max(answers)}")
        print(f"平均答案: {sum(answers) / len(answers):.1f}")
        print(f"操作符数量分布:")
        for i in range(1, max(num_operators) + 1):
            count = num_operators.count(i)
            if count > 0:
                percentage = count / len(num_operators) * 100
                print(f"  {i}个操作符: {count} ({percentage:.1f}%)")

def main():
    # 创建数据生成器
    generator = DatasetGenerator()
    
    # 生成适合小学生的数据集
    print("正在生成适合小学生的数学练习数据集...")
    train_data, test_data = generator.generate_dataset(
        train_size=1000,
        test_size=50,
        min_operators=1,
        max_operators=4  # 最多4个操作符，适合小学生
    )
    
    # 保存到文件
    print("保存训练数据...")
    generator.save_to_jsonl(train_data, "elementary_train_dataset.jsonl")
    
    print("保存测试数据...")
    generator.save_to_jsonl(test_data, "elementary_test_dataset.jsonl")
    
    print(f"\n数据集生成完成！")
    print(f"训练集: {len(train_data)} 条")
    print(f"测试集: {len(test_data)} 条")
    
    # 分析数据集
    print("\n训练集统计:")
    generator.analyze_dataset(train_data)
    
    print("\n测试集统计:")
    generator.analyze_dataset(test_data)
    
    # 显示几个示例
    print("\n训练集示例:")
    for i in range(5):
        sample = train_data[i]
        print(f"表达式: {sample['metadata']['expression']} = {sample['label']}")
    
    print("\n测试集示例:")
    for i in range(5):
        sample = test_data[i]
        print(f"表达式: {sample['metadata']['expression']} = {sample['label']}")

if __name__ == "__main__":
    main()
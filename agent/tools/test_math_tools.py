from typing import Dict, Any


class TestMathTools():
    """Test math tools, provides the following operations:
    - star_operation
    - flower_operation
    - moon_operation
    - sun_operation

    Args:
        avaialble_tools (list[str] | str, optional): Choose from the four operations
        or write 'all' for all tools. Defaults to None.
    """
    def __init__(self, avaialble_tools: tuple[str] | str = None):
        if avaialble_tools == "all":
            self.available_tools = (
                "star_operation", 
                "flower_operation", 
                "moon_operation", 
                "sun_operation"
            )
        else:
            self.available_tools = avaialble_tools
        assert self.available_tools, f"Should provide at least one tool."
        
    def star_operation(self, a: int, b: int) -> int:
        return a + b - 1
    
    def flower_operation(self, a: int, b: int) -> int:
        return a * 2 + b
    
    def moon_operation(self, a: int, b: int) -> int:
        return (a + b) * 2
    
    def sun_operation(self, a: int, b: int) -> int:
        return a * b - a

from abc import ABC, abstractmethod


class TransformTemplate(ABC):
    name: str = ""
    difficulty: str = ""

    @abstractmethod
    def generate_case(self) -> dict:
        ...

    def generate_example(self, n_demo: int = 3, n_test: int = 2) -> dict:
        cases = [self.generate_case() for _ in range(n_demo + n_test)]
        gold_code = cases[0]["gold_code"]
        return {
            "template_name": self.name,
            "difficulty": self.difficulty,
            "gold_code": gold_code,
            "demos": [{"input": c["input"], "output": c["output"]} for c in cases[:n_demo]],
            "tests": [{"input": c["input"], "output": c["output"]} for c in cases[n_demo:]],
        }

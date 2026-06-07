from abc import ABC, abstractmethod

from obli_flow.schema import ArtifactNode, StepRecord


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, record: StepRecord, action_id: str) -> list[ArtifactNode]:
        raise NotImplementedError

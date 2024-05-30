import torch

class ExplorationState:
    def __init__(
        self,
        sample_embed: torch.tensor,
    ):
        self.sample_embed = sample_embed.cpu()

    def validate(self):
        pass ## TODO

    def save(self, filename):
        state_dict = {
            "sample_embed": self.sample_embed.half(),
        }
        torch.save(
            state_dict,
            filename
        )
        print(f"Saved ExplorationState: {filename}")

    @classmethod
    def from_file(cls, filename: str):
        state_dict = torch.load(
            filename
        )
        return cls(
            sample_embed = state_dict["sample_embed"].float(),
        )
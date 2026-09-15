"""NBF with the locally optimal proposal. Kept as a class for the `nbf_op` method
spec and older call sites; it is `NeuralBayesianFilter(model, n, proposal=OptimalProposal())`."""

from embedex.distributions.embedding import FlowEmbedding

from nbf.filters.nbf import NeuralBayesianFilter
from nbf.filters.proposals import (  # noqa: F401  (re-exported for older imports)
    OPTIMAL_PROPOSAL_HOOKS,
    OptimalProposal,
    optimal_proposal_particle_set,
)


class OptimalProposalNeuralBayesianFilter(NeuralBayesianFilter):
    def __init__(self, model: FlowEmbedding, num_particles: int):
        super().__init__(model=model, num_particles=num_particles, proposal=OptimalProposal())

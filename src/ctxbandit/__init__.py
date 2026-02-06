from .policy_inference_single import PolicyInferenceSingle
from .policy_inference_joint import PolicyInferenceJoint
from .policy_inference_difference import PolicyInferenceDifference
from .utils import ReadableStrMixin

from importlib.metadata import version
__version__ = version("ctxbandit")

# The package `cvxpy` often raises a "Solution may be inaccurate" warning.
# It appears frequently but does not indicate any real issue in our usage, 
#     so we silence it here to avoid cluttering the output.
import warnings
warnings.filterwarnings(
    "ignore",
    message="Solution may be inaccurate",
    category=UserWarning,
    module="cvxpy.problems.problem"
)


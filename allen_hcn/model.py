from allennlp.models import Model


@Model.register("hcn")
class HybridCodeNetwork(Model):
    """
       This ``Model`` implements the Hybrid Code Network model described in:
       <https://github.com/voicy-ai/DialogStateTracking/tree/master/src/hcn>
    """
    pass
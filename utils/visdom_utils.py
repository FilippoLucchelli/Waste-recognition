"""Optional compatibility adapter for historical Visdom plots."""


class VisUtils:
    def __init__(self, metric_name, vis) -> None:
        self.metric_name = metric_name
        self.vis = vis
        self.plot = None

    def update_plot(self, epoch, metrics):
        values = [
            float(metric.detach().cpu()) if hasattr(metric, "detach") else float(metric)
            for metric in metrics
        ]
        update = None if self.plot is None else "append"
        self.plot = self.vis.line(
            X=[[epoch] * len(values)],
            Y=[values],
            win=self.plot,
            update=update,
            opts={
                "title": self.metric_name,
                "legend": ["validation", "training"],
            },
        )

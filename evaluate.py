from got10k.experiments import ExperimentOTB

from goturn import TrackerGOTURN

from got10k.trackers import Tracker


class IdentityTracker(Tracker):
    def __init__(self):
        super(IdentityTracker, self).__init__(name='IdentityTracker', is_deterministic=True)
    
    def init(self, image, box):
        self.box = box
    
    def update(self, image):
        
        return self.box


def main():
    tracker = TrackerGOTURN()
    # tracker = IdentityTracker()
    
    # GOT10k toolkit expects either extracted directories or zip files for
    # all sequences in OTB data directory.
    experiments = [ExperimentOTB('../../datasets/OTB_2013', version=2013),]
    
    for experiment in experiments:
        experiment.run(tracker, visualize=False)
        experiment.report([tracker.name])
        
    return 0

if __name__ == '__main__':
    import sys
    
    sys.exit(main())

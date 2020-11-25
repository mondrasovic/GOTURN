from got10k.experiments import ExperimentOTB

from goturn import TrackerGOTURN


def main():
    net_path = "../checkpoints/pytorch_goturn.pth.tar"
    tracker = TrackerGOTURN(net_path=net_path)
    
    # GOT10k toolkit expects either extracted directories or zip files for
    # all sequences in OTB data directory.
    experiments = [ExperimentOTB('../data/OTB', version=2013),]
    
    for experiment in experiments:
        experiment.run(tracker, visualize=False)
        experiment.report([tracker.name])
        
    return 0

if __name__ == '__main__':
    import sys
    
    sys.exit(main())

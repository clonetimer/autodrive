import argparse
import json
import _bootstrap
from src.baselines.random_agent import RandomAgent
from src.baselines.pid_controller import PIDController
from src.baselines.idm_controller import IDMController
from src.end_to_end.agent import EndToEndAgent
from src.navigation.learned_route_adapter import LearnedRouteAdapterAgent
from src.evaluation.evaluator import evaluate
from src.evaluation.learned_adapter_evaluator import evaluate as evaluate_with_adapter
from src.scenarios.generator import split_seeds
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description='Held-out closed-loop evaluation')
    parser.add_argument('--model')
    parser.add_argument('--baseline', choices=['random', 'pid', 'idm'])
    parser.add_argument('--config')
    parser.add_argument('--steering-adapter', help='optional learned residual steering adapter matched to --model SHA')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--split', choices=['validation', 'validation2', 'test', 'final', 'release', 'release2', 'release3', 'release4', 'release5', 'release6', 'validation3', 'validation4', 'validation5', 'validation6', 'release7', 'release8', 'release9', 'release10', 'validation7', 'release11', 'validation8', 'release12', 'validation9', 'release13', 'validation10', 'release14'], default='test')
    parser.add_argument('--output', required=True)
    parser.add_argument('--trajectories', action='store_true')
    args = parser.parse_args()
    if bool(args.model) == bool(args.baseline):
        parser.error('Choose exactly one of --model or --baseline')
    if args.model:
        policy = (LearnedRouteAdapterAgent(args.model, args.steering_adapter)
                  if args.steering_adapter else EndToEndAgent(args.model))
        config = load_config(args.config).env if args.config else policy.base.config.env if args.steering_adapter else policy.config.env
        factory = lambda name, seed: policy
        label = args.model + ('+learned-route-adapter' if args.steering_adapter else '')
    else:
        if not args.config:
            parser.error('--baseline requires --config')
        config = load_config(args.config).env
        if config.observation not in {'state', 'hybrid'} and args.baseline != 'random':
            parser.error('Rule baselines require state or hybrid observations')
        constructors = {'random': lambda seed: RandomAgent(seed), 'pid': lambda seed: PIDController(config.dt), 'idm': lambda seed: IDMController()}
        factory = lambda name, seed: constructors[args.baseline](seed + int(name.replace('agent', '')))
        label = args.baseline
    evaluator = evaluate_with_adapter if args.steering_adapter else evaluate
    result = evaluator(factory, config, split_seeds(args.split, args.episodes), args.output, label, args.trajectories)
    print(json.dumps(result['metrics'], indent=2))


if __name__ == '__main__':
    main()

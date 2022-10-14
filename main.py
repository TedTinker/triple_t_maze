#%%

from utils import args, new_text, already_done, folder
from train import Trainer
from itertools import accumulate

if(already_done):
    new_text("\n\n{} already done training and getting positions!".format(folder))

else:
    trainer = Trainer(args)
    trainer.train()
    
    new_text("\n\n{} finished training!".format(folder))

    import os 
    import torch
    from itertools import product
    from math import floor
    import enlighten

    load_names = os.listdir(folder + "/agents")
    load_names.sort()
    load_names = [l[6:-3] for l in load_names]
        
    pos_dict = {(l, a) : [] for l, a in product(load_names, ["1", "2", "3"])}

    pos_count = 10

    manager = enlighten.Manager()
    E = manager.counter(total = len(load_names), desc = "Positions:", unit = "ticks", color = "blue")
    epoch_sum = list(accumulate(trainer.args.epochs_per_arena))
    
    for load_name in load_names:
        positions_lists = []      
        trainer.new_load_name(folder, load_name)
        load_int = int(load_name)
        which_arena = 0 if load_int <= epoch_sum[0] else 1 if load_int <= epoch_sum[1] else 2
        next_arena = 1 if load_int == epoch_sum[0] else 2 if load_int == epoch_sum[1] else None
        if(next_arena == None): pass 
        else:
            trainer.current_arena = next_arena
            positions_list, arena_name = trainer.get_positions(size = pos_count)
            pos_dict[(load_name, arena_name)].append(positions_list)
        trainer.current_arena = floor(which_arena)
        positions_list, arena_name = trainer.get_positions(size = pos_count)
        pos_dict[(load_name, arena_name)].append(positions_list)
        E.update()
        
    torch.save((trainer.args.explore_type, pos_dict), folder + "/pos_dict.pt")
    torch.save(args, folder + "/args.pt")

    new_text("\n\n{} finished getting positions!".format(folder))

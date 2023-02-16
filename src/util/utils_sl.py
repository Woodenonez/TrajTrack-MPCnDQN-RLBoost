import sys
import json
import yaml
from typing import List

import numpy as np

from io import BufferedReader

#%% PGM files
def read_pgm(pgmf:BufferedReader, bit_depth:int=16, one_line_head:bool=False, skip_second_line:bool=True) -> List[list]:
    """Return a raster of integers from a PGM file as a list of lists."""
    """The head is normally [P5 Width Height Depth]"""
    header = pgmf.readline()  # the 1st line
    if one_line_head:
        magic_num = header[:2]
        (width, height) = [int(i) for i in header.split()[1:3]]
        depth = int(header.split()[3])
    else:
        magic_num = header
        if skip_second_line:
            comment = pgmf.readline() # the 2nd line if there is
            print(f'Comment: [{comment}]')
        (width, height) = [int(i) for i in pgmf.readline().split()]
        depth = int(pgmf.readline())

    if bit_depth == 8:
        assert magic_num[:2] == 'P5'
        assert depth <= 255
    elif bit_depth == 16:
        assert magic_num[:2] == b'P5'
        assert depth <= 65535

    raster = []
    for _ in range(height):
        row = []
        for _ in range(width):
            row.append(ord(pgmf.read(1)))
        raster.append(row)
    return raster

def read_pgm_and_process(pgmf:BufferedReader, inversed_pixel:bool, bit_depth:int=16, one_line_head:bool=False, skip_second_line:bool=True) -> np.ndarray:
    raw_map = read_pgm(pgmf, bit_depth, one_line_head, skip_second_line)
    the_map = np.array(raw_map)
    if inversed_pixel:
        the_map = 255 - the_map
    the_map[the_map>10] = 255
    the_map[the_map<=10] = 0
    the_map[:,[0,-1]] = 0
    the_map[[0,-1],:] = 0
    return the_map

#%% JSON for dynamic obstacles
'''
A stardard json file storing "predictions" should be:
{'info':[t1,x,y], 'pred_T1':[[a1,x1,y1,sx1,sy1], ..., [am,xm,ym,sxm,sym]], 'pred_T2':..., ...}
{'info':[t2,x,y], 'pred_T1':[[a1,x1,y1,sx1,sy1], ..., [am,xm,ym,sxm,sym]], 'pred_T2':..., ...} ...
One file for one object. Each row is a pred_t.

A stardard json file storing "trajectories" should be:
{'type':type, 'traj_x':[x1,x2,x3,...], 'traj_y':[y1,y2,y3,...]}
{'type':type, 'traj_x':[x1,x2,x3,...], 'traj_y':[y1,y2,y3,...]} ...
One file for multiple trajectories. Each row is a trajectory.
'''
def save_obj_as_json(obj_list, json_file_path):
    # pred_list: [pred_t1, pred_t2, ...]
    # traj_list: [traj1,   traj2,   ...]
    with open(json_file_path,'w+') as jf:
        for obj in obj_list:
            json.dump(obj, jf)
            jf.write('\n')

def read_obj_from_json(json_file): # return a list of dict
    obj_list = []
    with open(json_file,'r+') as jf:
        for obj in jf:
            try:
                obj_list.append(json.loads(obj))
            except: pass
    return obj_list

#%% YAML file S&L
def to_yaml(data, save_path, style=None):
    with open(save_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, default_style=style)
    print(f'Save to {save_path}.')
    return 0

def to_yaml_all(data_list, save_path, style=None):
    with open(save_path, 'w') as f:
        yaml.dump_all(data_list, f, explicit_start=True, default_flow_style=False, default_style=style)
    print(f'Save to {save_path}.')
    return 0

def from_yaml(load_path, vb=True):
    with open(load_path, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
            if vb:
                print(f'Load from {load_path}.')
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml

def from_yaml_all(load_path, vb=True):
    with open(load_path, 'r') as stream:
        parsed_yaml_list = []
        try:
            for data in yaml.load_all(stream, Loader=yaml.FullLoader):
                parsed_yaml_list.append(data)
            if vb:
                print(f'Load from {load_path}.')
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml_list



import yaml

""" 
    File that contains all the neccessary configuration parameters for the 
    MPC Trajectory Generation Module
"""

class Configurator:
    def __init__(self, yaml_fp):
        self.__prtname = '[MPC-CFG]'
        print(f'{self.__prtname} Loading configuration from "{yaml_fp}".')
        with open(yaml_fp, 'r') as stream:
            yaml_load = yaml.safe_load(stream)
        for key in yaml_load:
            setattr(self, key, yaml_load[key])
            # getattr(self, key).__set_name__(self, key)
        print(f'{self.__prtname} Configuration done.')

if __name__ == '__main__':
    yaml_fpath = '/home/ze/Documents/Code2/DyObAv_MPCnEBM_Assemble/config/mpc_default.yaml'
    cfg = Configurator(yaml_fpath)
    print(cfg)


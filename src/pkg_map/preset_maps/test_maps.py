import math


def return_test_map(index:int):
    if index==1:
        return _return_single_square_map()
    elif index==2:
        return _return_simple_zigzag_map()
    elif index==3:
        return _return_multi_zigzag_map()
    elif index==4:
        return _return_narrow_corridor_map()
    elif index==5:
        return _return_yshape_obstacle_map()
    elif index==6:
        return _return_sharp_turn_map()
    else:
        raise ValueError("Invalid index")

def return_test_map_dyanmic():
    return _return_classic_alpha_map()

### With only static obstacles
def _return_single_square_map():
    boundary_coords = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    obstacle_list = [ [(3.0, 3.0), (3.0, 7.0), (7.0, 7.0), (7.0, 3.0)] ]
    start = (1.0, 1.0, math.radians(0))
    end = (8.0, 8.0, math.radians(90))
    return boundary_coords, obstacle_list, start, end

def _return_simple_zigzag_map():
    boundary_coords = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]
    obstacle_list = [
        [(5.0, 0.0), (5.0, 15.0), (7.0, 15.0), (7.0, 0.0)],
        [(15.0, 20.0), (15.0, 5.0), (13.0, 5.0), (13.0, 20.0)]
    ]
    start = (1.0, 1.0, math.radians(0))
    end = (18.0, 18.0, math.radians(90))
    return boundary_coords, obstacle_list, start, end

def _return_multi_zigzag_map():
    boundary_coords = [(3.0, 58.0), (3.0, 3.0), (58.0, 3.0), (58.0, 58.0)]
    obstacle_list = [
        [(21.1, 53.1), (21.4, 15.1), (9.3, 15.1), (9.1, 53.1)],
        [(35.7, 52.2), (48.2, 52.3), (48.7, 13.6), (36.1, 13.8)], 
        [(17.0, 50.5),(30.7, 50.3), (30.6, 45.0), (17.5, 45.1)],
        [(26.4, 39.4), (40.4, 39.3), (40.5, 35.8), (26.3, 36.0)],
        [(19.3, 31.7), (30.3, 31.6), (30.1, 27.7), (18.9, 27.7)],
        [(26.9, 22.7), (41.4, 22.6), (41.1, 17.5), (27.4, 17.6)] ]
    start = (30.0, 5.0, math.radians(90))
    end = (30.0, 55.0, math.radians(90))
    return boundary_coords, obstacle_list, start, end

def _return_narrow_corridor_map():
    boundary_coords = [(40.0, 58.0), (7.5, 58.0), (7.5, 18.0), (40.0, 18.0)]
    obstacle_list = [
        [(14.0, 57.6), (42.1, 57.6), (42.2, 52.0), (13.4, 52.0)], 
        [(7.7, 49.1), (32.2, 49.0), (32.1, 45.3), (7.7, 45.8)], 
        [(34.2, 53.0), (41.2, 53.1), (40.9, 31.7), (34.4, 31.9)], 
        [(35.7, 41.7), (35.7, 36.8), (11.7, 39.8), (12.1, 44.0), (31.3, 43.3)], 
        [(5.8, 37.6), (24.1, 35.0), (23.6, 29.8), (5.0, 31.8)], 
        [(27.1, 39.7), (32.7, 39.0), (32.8, 24.7), (16.2, 20.9), (14.5, 25.9), (25.3, 26.7), (27.9, 31.4), (26.1, 39.2)] ]
    start = (10.3, 55.8, math.radians(270))
    end = (38.1, 25.0, math.radians(300))
    return boundary_coords, obstacle_list, start, end

def _return_yshape_obstacle_map():
    boundary_coords = [(-1.0, 0.0), (15.0, 0.0), (15.0, 18.0), (-1.0, 18.0)]
    obstacle_list = [[(5.0, 0.0), (8.0,0.0), (8.0,8.0), (12.0,12.0), 
                      (10.0,13.0), (6.5,9.0), (5.0,13.0), (3.0,12.0), (5.0,8.0)]]
    start = (2.0, 2.0, math.radians(90))
    end = (10.0, 2.0, math.radians(275))
    return boundary_coords, obstacle_list, start, end

def _return_sharp_turn_map():
    boundary_coords = [(0.0, 0.0), (15.0, 0.0), (15.0, 15.0), (0.0, 15.0)]
    obstacle_list = [[(7.0, 0.0), (7.0, 7.0), (8.0, 12.0), (9.0, 7.0), (9.0, 0.0)]]
    start = (5.0, 2.0, math.radians(90))
    end = (11.0, 2.0, math.radians(265))
    return boundary_coords, obstacle_list, start, end

### With dynamic obstacles
def _return_classic_alpha_map():
    """Classic Alpha map (with dynamic obstacles) from the first paper, CASE2021"""
    boundary_coords = [(11.9, 3.6), (11.9, 50.6), (47.3, 50.6), (47.3, 3.6)]
    obstacle_list = [[(11.9, 11.8), (22.2, 11.8), (22.2, 15.9), (11.9, 15.9)],
        [(11.9, 20.4), (22.2, 20.4), (22.2, 25.0), (11.9, 25.0)],
        [(28.0, 25.5), (28.0, 20.5), (32.4, 20.5), (32.4, 15.7), (28.0, 15.7), (28.0, 3.6), (37.8, 3.6), (37.8, 25.5)], # low
        [(15.9, 29), (37.7, 29), (37.7, 44.5), (25.3, 44.5), (25.3, 40.7), (35.0, 40.7), (35.0, 31.7), (15.9, 31.7)], # up
        [(29.8, 28.7), (29.8, 25.8), (34.5, 25.8), (34.5, 28.7)] ]
    start = (18.9, 7.0, math.radians(45))
    end = (44.7, 6.8, math.radians(270))
    # Starting point, ending point, freq, x radius, y radius
    reciprocating_dyn_obs_list = [
        [[18.5, 18.2],[28.1, 18.2], 0.06, 0.5, 1.0],
        [[16.775, 34.0], [22.5, 42.2], 0.07, 0.3, 0.7],
        [[44.3, 9.2], [40.5, 31.8], 0.0745, 0.6, 0.6]
    ]
    return boundary_coords, obstacle_list, start, end, reciprocating_dyn_obs_list




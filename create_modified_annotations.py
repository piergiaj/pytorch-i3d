"""
CSV structure:
    <sample id>;<sample label>;<action label>;<category label>
 
Notes:
    * We use the term "category" and "scene" interchangeably
    * There are a total of 5 action labels (swiping-left, swiping-right, swiping-down, swiping-up, other)
    * There are a total of 2 category labels (swiping, other)
"""

import os
import sys

src = './data/jester/labels/'

for in_name in ['jester-v1-train.csv', 'jester-v1-validation.csv']:
    with open(src + in_name) as f_in:
        out_name = src + in_name.split('.csv')[0] + '-modified' + '.csv'
        if os.path.exists(out_name):
            os.remove(out_name)
        f_out = open(out_name, 'w+')
        for line in f_in:
            line = line.strip('\n')
            f_out.write(line)
            action, category = None, None
            line = line.lower()
            if 'swiping' in line:
                if 'left' in line:
                    action = 'swiping-left'
                elif 'right' in line:
                    action = 'swiping-right'
                elif 'down' in line:
                    action = 'swiping-down'
                elif 'up' in line:
                    action = 'swiping-up'
                category = 'swiping'
            else:
                action = 'other'
                category = 'other'

            f_out.write(';{};{}\n'.format(action, category))

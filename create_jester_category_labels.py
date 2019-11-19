"""
Category labels (which serve as 'scene' classes, as opposed to 'action' classes which are the original labels):
  * swiping-left-right
  * swiping-up-down
  * pushing-pulling-hand
  * pushing-pulling-two-fingers
  * sliding-left-right
  * sliding-up-down
  * rolling-hand
  * turning-hand
  * zooming-full-hand
  * zooming-two-fingers
  * thumb
"""

import os
import sys

src = './data/jester/labels/'

for f_name in ['jester-v1-train.csv', 'jester-v1-validation.csv']:
  with open(src + f_name) as f_in:
    f_out = open(src + f_name.split('.csv')[0] + '-modified' + '.csv', 'w+')
    for line in f_in:
      line = line.strip('\n')
      f_out.write(line)
      category = None
      line = line.lower()
      if 'swiping' in line:
        category = 'swiping-left-right' if 'left' in line or 'right' in line else 'swiping-up-down'
      if 'pushing' in line or 'pulling' in line:
        category = 'pushing-pulling-hand' if 'hand' in line else 'pushing-pulling-two-fingers'
      if 'sliding' in line:
        category = 'sliding-left-right' if 'left' in line or 'right' in line else 'sliding-up-down'
      if 'rolling' in line:
        category = 'rolling-hand'
      if 'turning' in line:
        category = 'turning-hand'
      if 'zooming' in line:
        category = 'zooming-full-hand' if 'full' in line else 'zooming-two-fingers'
      if 'thumb' in line:
        category = 'thumb'
      
      if category:
        f_out.write(';' + category)
      f_out.write('\n')

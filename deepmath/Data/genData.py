import csv
import os
import subprocess
import time


with open('train.csv', 'rb') as train:
    trainreader = csv.reader(train, delimiter=',')
    for row in trainreader:
        idx, eqn = row
        filename = 'eqn{}.tex'.format(idx)
        content = [r'\documentclass[preview]{standalone}', 
                r'\begin{document}', '${}$'.format(eqn), r'\end{document}']

        content = '\n'.join(content)
        with open(filename, 'w') as f:
            f.write(content)

        cmd = ['pdflatex', filename]
        proc = subprocess.Popen(cmd)
        proc.communicate()
        
        base = 'eqn{}'.format(idx)
        
        convert_cmd = ['convert', '-density', '300', '{}.pdf'.format(base), '-quality', '90', 'Images/{}.png'.format(base)]
        subprocess.call(convert_cmd)

        os.unlink(base + '.log')
        os.unlink(base + '.aux')
        os.unlink(base + '.tex')
        os.unlink(base + '.pdf')




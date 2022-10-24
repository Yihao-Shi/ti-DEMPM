def MonitorMPM(mpm, printNum, t, step, OutputPath):
    global filename
    print('---------------------', 'Writing MPM Graphic files ', printNum, '---------------------')
    if printNum < 10:
        filename = OutputPath + '/MonitorMPM' + str(printNum) + '.asc'
    elif printNum < 100:
        filename = OutputPath + '/MonitorMPM' + str(printNum) + '.asc'
    elif printNum < 1000:
        filename = OutputPath + '/MonitorMPM' + str(printNum) + '.asc'
    elif printNum < 10000:
        filename = OutputPath + '/MonitorMPM' + str(printNum) + '.asc'
    elif printNum < 100000:
        filename = OutputPath + '/MonitorMPM' + str(printNum) + '.asc'

    monitorFile = open(filename, 'w')
    monitorFile.write('# MPM Program-State Information #\n')
    monitorFile.write('MPM Programming Time = ' + str(t) + ' Step = ' + str(step) + ' ' + filename + ' \n')

    monitorFile.write('\n############################################\n')
    monitorFile.write('\n##                BODYID                  ##\n')
    monitorFile.write('\n############################################\n')
    monitorFile.write(str(mpm.particleSize) + '\n')
    for np in range(mpm.particleSize):
        for i in range(10):
            monitorFile.write(str(mpm.lp.bodyID[np]) + ' ')
        monitorFile.write('\n')

    monitorFile.write('\n############################################\n')
    monitorFile.write('\n##                BODYID                  ##\n')
    monitorFile.write('\n############################################\n')
    monitorFile.write(str(mpm.particleSize) + '\n')
    for np in range(mpm.particleSize):
        for i in range(10):
            monitorFile.write(str(mpm.lp.bodyID[np]) + ' ')
        monitorFile.write('\n')

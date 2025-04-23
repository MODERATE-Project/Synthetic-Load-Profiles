import marimo

__generated_with = "0.10.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import io
    from io import StringIO
    import csv
    import gzip
    import torch
    import pandas as pd

    from main import run
    return StringIO, csv, gzip, io, mo, pd, run, torch


@app.cell
def _(StringIO, csv, pd):
    def read_csv(file):
        str_ = str(file.contents(), 'utf-8')
        data = StringIO(str_)
        sep = csv.Sniffer().sniff(data.getvalue()).delimiter
        df = pd.read_csv(data, sep = sep)
        df = df.set_index(df.columns[0])
        return df


    #def get_model_state(modelFile):
    #    buffer = io.BytesIO(modelFile.value[0].contents)
    #    with gzip.GzipFile(fileobj = buffer) as file:
    #        modelState = torch.load(file, weights_only = False)
    #    return modelState


    def to_boolean(str_):
        bool_dict = {'yes': True, 'no': False, 'on': True, 'off': False}
        return bool_dict[str_]
    return read_csv, to_boolean


@app.cell
def _(mo):
    # Basic tab
    appTitle = mo.md("### **GAN**")
    modelType = mo.ui.dropdown(options = ['GAN', 'WGAN'], value = 'GAN', label = 'Model type:')
    projectName = mo.ui.text(label = 'Project name:')
    inputFileLabel = mo.md('Input data:')
    inputFile = mo.ui.file(label = 'Upload', filetypes = ['.csv'])
    outputFormat = mo.ui.dropdown(options = ['.npy', '.csv', '.xslx'], value = '.npy', label = 'Output file format:')
    logStats = mo.ui.radio(options = ['no', 'yes'], value = 'yes', inline = True, label = 'Log stats:')
    useWandb = mo.ui.radio(options = ['off', 'on'], value = 'off', inline = True, label = 'Wandb:')
    epochCount = mo.ui.number(start = 1, step = 1, value = 100, label = 'Number of epochs:')
    saveFreq = mo.ui.number(start = 1, step = 1, value = 10, label = 'Save frequency:')
    saveModels = mo.ui.radio(options = ['no', 'yes'], value = 'no', inline = True, label = 'Save models:')
    savePlots = mo.ui.radio(options = ['no', 'yes'], value = 'yes', inline = True, label = 'Save plots:')
    saveSamples = mo.ui.radio(options = ['no', 'yes'], value = 'no', inline = True, label = 'Save samples:')
    checkForMinStats = mo.ui.number(start = 1, step = 1, value = 100, label = 'Check for improving metric after this epoch:')
    # Work with existing model
    modelFileLabel = mo.md('Model:')
    modelFile = mo.ui.file_browser(label = 'Upload', multiple = False)
    createData = mo.ui.radio(options = ['yes', 'no'], value = 'yes', inline = True, label = 'Create data ¹:')
    createDataFootnote = mo.md('<div style="text-align: right">¹ <sub>If no, continue training</sub></div>')

    # Advanced tab
    batchSize = mo.ui.number(start = 1, step = 1, value = 40, label = 'Batch size:')
    lrGen = mo.ui.text(label = 'Generator learning rate:', value = str(1e-4/3.25))
    lrDis = mo.ui.text(label = 'Discriminator learning rate:', value = str(1e-4/2.25))
    loopCountGen = mo.ui.number(start = 1, step = 1, value = 1, label = 'Generator loop count:')

    # Start
    button = mo.ui.run_button(label = 'Start')
    return (
        appTitle,
        batchSize,
        button,
        checkForMinStats,
        createData,
        createDataFootnote,
        epochCount,
        inputFile,
        inputFileLabel,
        logStats,
        loopCountGen,
        lrDis,
        lrGen,
        modelFile,
        modelFileLabel,
        modelType,
        outputFormat,
        projectName,
        saveFreq,
        saveModels,
        savePlots,
        saveSamples,
        useWandb,
    )


@app.cell
def _(
    batchSize,
    checkForMinStats,
    createData,
    createDataFootnote,
    epochCount,
    inputFile,
    inputFileLabel,
    logStats,
    loopCountGen,
    lrDis,
    lrGen,
    mo,
    modelFile,
    modelFileLabel,
    modelType,
    outputFormat,
    projectName,
    saveFreq,
    saveModels,
    savePlots,
    saveSamples,
    useWandb,
):
    basicTab = mo.vstack([
        modelType,
        projectName,
        mo.hstack([
            inputFileLabel,
            inputFile,
        ], justify = 'start'),
        outputFormat,
        logStats,
        useWandb,
        epochCount,
        saveFreq,
        saveModels,
        savePlots,
        saveSamples,
        checkForMinStats,
        mo.md('---'),
        mo.accordion({
            '**Work with existing model**': mo.vstack([
                mo.hstack([
                    modelFileLabel,
                    modelFile
                ], justify = 'start'),
                mo.hstack([
                    createData,
                    createDataFootnote
                ])
            ])
        })
    ])

    advancedTab = mo.vstack([
        batchSize,
        lrGen,
        lrDis,
        loopCountGen,
        mo.md('---')
    ])

    tabs = mo.ui.tabs({
        '🔧 Basic': basicTab,
        '🛠️ Advanced': advancedTab
    })
    return advancedTab, basicTab, tabs


@app.cell
def _(appTitle, button, mo, tabs):
    mo.vstack([
        appTitle,
        tabs,
        button
    ])
    return


@app.cell
def _(
    batchSize,
    button,
    checkForMinStats,
    createData,
    epochCount,
    inputFile,
    logStats,
    loopCountGen,
    lrDis,
    lrGen,
    mo,
    modelFile,
    modelType,
    outputFormat,
    pd,
    projectName,
    read_csv,
    run,
    saveFreq,
    saveModels,
    savePlots,
    saveSamples,
    to_boolean,
    useWandb,
):
    if button.value:
        with mo.redirect_stdout():
            print('Processing...')
            if modelType.value == 'GAN':
                from model.GAN_params import params
            elif modelType.value == 'WGAN':
                from model.WGAN_params import params
            params['outputFormat'] = outputFormat.value
            params['epochCount'] = epochCount.value
            params['saveFreq'] = saveFreq.value
            params['saveModels'] = to_boolean(saveModels.value)
            params['savePlots'] = to_boolean(savePlots.value)
            params['saveSamples'] = to_boolean(saveSamples.value)
            params['checkForMinStats'] = checkForMinStats.value
            params['batchSize'] = batchSize.value
            params['lrGen'] = float(lrGen.value)
            params['lrDis'] = float(lrDis.value)
            params['genLoopCount'] = loopCountGen.value
            if inputFile.value:
                inputFileProcd = read_csv(inputFile)
                inputFileProcd.index = pd.to_datetime(inputFileProcd.index, format = 'mixed')
            else:
                inputFileProcd = None
            run(
                params = params,
                modelType = modelType.value,
                projectName = projectName.value,
                inputFile = inputFileProcd,
                logStats = to_boolean(logStats.value),
                useWandb = to_boolean(useWandb.value),
                modelPath = modelFile.value[0].id,
                createData = to_boolean(createData.value),
                useMarimo = True
            )
            print('Done!')
    return inputFileProcd, params


if __name__ == "__main__":
    app.run()

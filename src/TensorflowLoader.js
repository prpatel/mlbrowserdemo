import * as tf from "@tensorflow/tfjs/dist/index";

let indexFrom, maxLen, wordIndex, model;
const HOSTED_URLS = {
    model:
        'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json',
    metadata:
        'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json'
};

const LOCAL_URLS = {
    model: 'http://localhost:3000/model.json',
    metadata: 'http://localhost:3000/metadata.json'
};

async function initModels() {
    console.log('initModels');
    let model = await loadHostedPretrainedModel(HOSTED_URLS.model);
    await loadMetadata(HOSTED_URLS);
    return model;
}

async function loadHostedPretrainedModel(url) {
    console.log('loadHostedPretrainedModel');
    model = await tf.loadModel(url);
    return model;
}

async function loadHostedMetadata(url) {
    console.log('loadHostedMetadata');
    try {
        const metadataJson = await fetch(url);
        const metadata = await metadataJson.json();
        return metadata;
    } catch (err) {
        console.error(err);
    }
}

async function loadMetadata(urls) {
    console.log('loadMetadata');
    const sentimentMetadata = await loadHostedMetadata(urls.metadata);
    // ui.showMetadata(sentimentMetadata);
    indexFrom = sentimentMetadata['index_from'];
    maxLen = sentimentMetadata['max_len'];
    console.log('indexFrom = ' + indexFrom);
    console.log('maxLen = ' + maxLen);
    wordIndex = sentimentMetadata['word_index']
    return sentimentMetadata;
}

function predict(text) {
    // Convert to lower case and remove all punctuations.
    const inputText =
        text.trim().toLowerCase().replace(/(\.|,|!)/g, '').split(' ');
    // Look up word indices.
    const inputBuffer = tf.buffer([1, maxLen], 'float32');
    for (let i = 0; i < inputText.length; ++i) {
        // TODO(cais): Deal with OOV words.
        const word = inputText[i];
        inputBuffer.set(wordIndex[word] + indexFrom, 0, i);
    }
    const input = inputBuffer.toTensor();

    // ui.status('Running inference');
    const beginMs = performance.now();
    const predictOut = model.predict(input);
    const score = predictOut.dataSync()[0];
    predictOut.dispose();
    const endMs = performance.now();
    console.log(score)
    return {score: score, elapsed: (endMs - beginMs)};
}

export {initModels, predict};
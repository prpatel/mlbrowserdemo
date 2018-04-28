import React, {Component} from 'react';
import './App.css';
import { Button } from 'react-bootstrap';
import { PageHeader } from 'react-bootstrap';
import { Panel } from 'react-bootstrap';
import { ProgressBar } from 'react-bootstrap';
import { Label } from 'react-bootstrap';
import * as tf from './TensorflowLoader';

class App extends Component {
    constructor(props, context) {
        super(props, context);

        this.handleClick = this.handleClick.bind(this);
        this.getPredictionScore = this.getPredictionScore.bind(this);

        this.state = {
            modelLoaded: false,
            score: 0,
            elapsed: 0
        };
    }
    handleClick() {
        const predictor = tf.initModels();
        this.setState({ modelLoaded: true });

    }

    getPredictionScore(event) {
        this.setState(tf.predict(event.target.value))
    }

    render() {

        const { modelLoaded } = this.state;

        return (
<div className="App">
    <PageHeader >
        <h1 className="App-title">Text Sentiment Analysis</h1>
        Convolutional Neural Network trained against the IMDB Reviews Dataset
    </PageHeader>

        <Button
            bsStyle="primary"
            disabled={modelLoaded}
            onClick={this.handleClick}
        >Load Trained Model
        </Button>

    <p></p>


    <Panel className="textPanel">
        <Panel.Heading>
            <Panel.Title componentClass="h3">Input text below to get a score</Panel.Title>
        </Panel.Heading>
        <Panel.Body><textarea className="textInput" onChange={this.getPredictionScore}></textarea></Panel.Body>
    </Panel>

    <Panel className="textPanel">
        <Panel.Heading>
            <Panel.Title componentClass="h3">Sentiment Analysis</Panel.Title>
        </Panel.Heading>
        <Panel.Body>
            <h2>
                Score <Label>{this.state.score.toFixed(3)}</Label>
            </h2>
            <h2>
                Time <Label>{this.state.elapsed.toFixed(3)} ms</Label>
            </h2>

            <ProgressBar now={this.state.score * 100} />;
        </Panel.Body>
    </Panel>
</div>
        );
    }
}

export default App;

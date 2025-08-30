import React, { useState, useEffect } from 'react';
import './App.css';
import axios from 'axios';
import { Upload, Brain, BarChart3, Settings, Download, Play, Eye, FileText, AlertTriangle, CheckCircle, Loader, TrendingUp, Database } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Textarea } from './components/ui/textarea';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Badge } from './components/ui/badge';
import { Progress } from './components/ui/progress';
import { Alert, AlertDescription } from './components/ui/alert';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select';
import { useToast } from './hooks/use-toast';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [activeTab, setActiveTab] = useState('upload');
  const [datasets, setDatasets] = useState([]);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [trainingStatus, setTrainingStatus] = useState({ status: 'idle', progress: 0, message: '' });
  const [edaResults, setEdaResults] = useState({});
  const [predictionForm, setPredictionForm] = useState({
    hyp: '',
    tgt: '',
    src: '',
    task: 'DM'
  });
  const [predictionResult, setPredictionResult] = useState(null);
  const [modelStats, setModelStats] = useState({ models_trained: 0, model_ready: false });
  const { toast } = useToast();

  // Load initial data
  useEffect(() => {
    loadDatasets();
    loadModelStats();
  }, []);

  // Poll training status
  useEffect(() => {
    let interval;
    if (trainingStatus.status === 'training' || trainingStatus.status === 'processing') {
      interval = setInterval(() => {
        checkTrainingStatus();
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [trainingStatus.status]);

  const loadDatasets = async () => {
    try {
      const response = await axios.get(`${API}/datasets`);
      setDatasets(response.data);
    } catch (error) {
      console.error('Error loading datasets:', error);
    }
  };

  const loadModelStats = async () => {
    try {
      const response = await axios.get(`${API}/model-stats`);
      setModelStats(response.data);
    } catch (error) {
      console.error('Error loading model stats:', error);
    }
  };

  const checkTrainingStatus = async () => {
    try {
      const response = await axios.get(`${API}/training-status`);
      setTrainingStatus(response.data);
      if (response.data.status === 'completed' || response.data.status === 'error') {
        loadModelStats();
        if (response.data.status === 'completed') {
          toast({
            title: "Training Completed!",
            description: `Ensemble trained successfully with ${response.data.current_accuracy?.toFixed(3)} accuracy`,
          });
        }
      }
    } catch (error) {
      console.error('Error checking training status:', error);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API}/upload-dataset`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      setUploadedFiles([...uploadedFiles, response.data]);
      loadDatasets();
      
      toast({
        title: "File Uploaded Successfully!",
        description: `${file.name} has been processed and is ready for analysis.`,
      });
    } catch (error) {
      toast({
        title: "Upload Failed",
        description: error.response?.data?.detail || "Failed to upload file",
        variant: "destructive",
      });
    }
  };

  const performEDA = async (datasetId) => {
    try {
      const response = await axios.post(`${API}/eda/${datasetId}`);
      setEdaResults(prev => ({ ...prev, [datasetId]: response.data }));
      setActiveTab('eda');
      
      toast({
        title: "EDA Completed!",
        description: "Exploratory data analysis has been generated.",
      });
    } catch (error) {
      toast({
        title: "EDA Failed",
        description: error.response?.data?.detail || "Failed to perform EDA",
        variant: "destructive",
      });
    }
  };

  const startTraining = async () => {
    const labeledDatasets = datasets.filter(d => d.columns.includes('label')).map(d => d.id);
    
    if (labeledDatasets.length === 0) {
      toast({
        title: "No Training Data",
        description: "Please upload datasets with labels to train the model.",
        variant: "destructive",
      });
      return;
    }

    try {
      await axios.post(`${API}/train-models`, {
        dataset_ids: labeledDatasets,
        n_models: 5,
        test_split: 0.2,
        random_seed: 42
      });
      
      setTrainingStatus({ status: 'training', progress: 0, message: 'Training started...' });
      setActiveTab('train');
      
      toast({
        title: "Training Started",
        description: "Ensemble model training has begun. This may take several minutes.",
      });
    } catch (error) {
      toast({
        title: "Training Failed",
        description: error.response?.data?.detail || "Failed to start training",
        variant: "destructive",
      });
    }
  };

  const handlePrediction = async () => {
    if (!modelStats.model_ready) {
      toast({
        title: "Model Not Ready",
        description: "Please train the model first before making predictions.",
        variant: "destructive",
      });
      return;
    }

    try {
      const response = await axios.post(`${API}/predict`, predictionForm);
      setPredictionResult(response.data);
      
      toast({
        title: "Prediction Complete",
        description: `Predicted: ${response.data.predicted_type} (${response.data.severity_percent}% severity)`,
      });
    } catch (error) {
      toast({
        title: "Prediction Failed",
        description: error.response?.data?.detail || "Failed to make prediction",
        variant: "destructive",
      });
    }
  };

  const handleBatchPredict = async (datasetId) => {
    if (!modelStats.model_ready) {
      toast({
        title: "Model Not Ready",
        description: "Please train the model first before making predictions.",
        variant: "destructive",
      });
      return;
    }

    try {
      const response = await axios.post(`${API}/batch-predict/${datasetId}`);
      
      toast({
        title: "Batch Prediction Complete",
        description: `Generated ${response.data.total_predictions} predictions. Click download to get results.`,
      });
    } catch (error) {
      toast({
        title: "Batch Prediction Failed",
        description: error.response?.data?.detail || "Failed to perform batch prediction",
        variant: "destructive",
      });
    }
  };

  const downloadResults = (datasetId) => {
    window.open(`${API}/download-results/${datasetId}`, '_blank');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Brain className="h-12 w-12 text-indigo-600" />
            <h1 className="text-4xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
              HalluciNet Detector
            </h1>
          </div>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Advanced Deep Learning Pipeline for Detecting and Classifying Hallucinations in LLM-Generated Outputs
          </p>
          
          {/* Status Bar */}
          <div className="flex items-center justify-center gap-6 mt-6 p-4 bg-white/60 backdrop-blur-sm rounded-lg border border-white/20">
            <div className="flex items-center gap-2">
              <Database className="h-5 w-5 text-blue-500" />
              <span className="text-sm font-medium">{datasets.length} Datasets</span>
            </div>
            <div className="flex items-center gap-2">
              <Brain className="h-5 w-5 text-green-500" />
              <span className="text-sm font-medium">{modelStats.models_trained} Models Trained</span>
            </div>
            <div className="flex items-center gap-2">
              {modelStats.model_ready ? (
                <CheckCircle className="h-5 w-5 text-green-500" />
              ) : (
                <AlertTriangle className="h-5 w-5 text-amber-500" />
              )}
              <span className="text-sm font-medium">
                {modelStats.model_ready ? 'Ready for Predictions' : 'Training Required'}
              </span>
            </div>
          </div>
        </div>

        {/* Main Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-5 mb-8 bg-white/50 backdrop-blur-sm">
            <TabsTrigger value="upload" className="flex items-center gap-2">
              <Upload className="h-4 w-4" />
              Upload Data
            </TabsTrigger>
            <TabsTrigger value="eda" className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              EDA
            </TabsTrigger>
            <TabsTrigger value="train" className="flex items-center gap-2">
              <Settings className="h-4 w-4" />
              Train Models
            </TabsTrigger>
            <TabsTrigger value="predict" className="flex items-center gap-2">
              <Brain className="h-4 w-4" />
              Predict
            </TabsTrigger>
            <TabsTrigger value="batch" className="flex items-center gap-2">
              <FileText className="h-4 w-4" />
              Batch Process
            </TabsTrigger>
          </TabsList>

          {/* Upload Tab */}
          <TabsContent value="upload" className="space-y-6">
            <Card className="bg-white/70 backdrop-blur-sm border-white/20">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Upload className="h-6 w-6 text-indigo-600" />
                  Dataset Upload
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="border-2 border-dashed border-indigo-200 rounded-lg p-8 text-center hover:border-indigo-300 transition-colors">
                    <Upload className="h-12 w-12 text-indigo-400 mx-auto mb-4" />
                    <label htmlFor="file-upload" className="cursor-pointer">
                      <span className="text-lg font-medium text-indigo-600 hover:text-indigo-700">
                        Click to upload CSV files
                      </span>
                      <p className="text-sm text-gray-500 mt-2">
                        Supports train, validation, and test datasets
                      </p>
                    </label>
                    <input
                      id="file-upload"
                      type="file"
                      accept=".csv"
                      onChange={handleFileUpload}
                      className="hidden"
                    />
                  </div>
                  
                  {/* Uploaded Datasets */}
                  {datasets.length > 0 && (
                    <div className="space-y-2">
                      <h3 className="font-medium text-gray-900">Uploaded Datasets</h3>
                      <div className="grid gap-3">
                        {datasets.map((dataset) => (
                          <div key={dataset.id} className="p-4 bg-white rounded-lg border border-gray-200 shadow-sm">
                            <div className="flex items-center justify-between">
                              <div>
                                <h4 className="font-medium text-gray-900">{dataset.filename}</h4>
                                <p className="text-sm text-gray-500">
                                  {dataset.rows} rows, {dataset.columns.length} columns
                                </p>
                                <div className="flex gap-1 mt-2">
                                  {dataset.columns.includes('label') && (
                                    <Badge variant="secondary">Labeled</Badge>
                                  )}
                                  {dataset.columns.includes('p(Hallucination)') && (
                                    <Badge variant="outline">Has Probability</Badge>
                                  )}
                                </div>
                              </div>
                              <Button
                                onClick={() => performEDA(dataset.id)}
                                variant="outline"
                                size="sm"
                                className="flex items-center gap-1"
                              >
                                <Eye className="h-4 w-4" />
                                Analyze
                              </Button>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* EDA Tab */}
          <TabsContent value="eda" className="space-y-6">
            <Card className="bg-white/70 backdrop-blur-sm border-white/20">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-6 w-6 text-indigo-600" />
                  Exploratory Data Analysis
                </CardTitle>
              </CardHeader>
              <CardContent>
                {Object.keys(edaResults).length === 0 ? (
                  <div className="text-center py-8">
                    <BarChart3 className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                    <p className="text-gray-500">No analysis results yet. Upload data and click "Analyze" to get started.</p>
                  </div>
                ) : (
                  <div className="space-y-8">
                    {Object.entries(edaResults).map(([datasetId, results]) => (
                      <div key={datasetId} className="space-y-6">
                        <h3 className="text-lg font-semibold text-gray-900 border-b pb-2">
                          Dataset Analysis: {datasets.find(d => d.id === datasetId)?.filename}
                        </h3>
                        
                        {/* Statistics Cards */}
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                          {results.class_distribution && Object.keys(results.class_distribution).length > 0 && (
                            <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
                              <h4 className="font-medium text-blue-900 mb-2">Label Distribution</h4>
                              {Object.entries(results.class_distribution).map(([label, count]) => (
                                <div key={label} className="flex justify-between text-sm">
                                  <span className="text-blue-700">{label}:</span>
                                  <span className="font-medium text-blue-900">{count}</span>
                                </div>
                              ))}
                            </div>
                          )}
                          
                          {results.type_distribution && (
                            <div className="p-4 bg-green-50 rounded-lg border border-green-200">
                              <h4 className="font-medium text-green-900 mb-2">Task Distribution</h4>
                              {Object.entries(results.type_distribution).map(([task, count]) => (
                                <div key={task} className="flex justify-between text-sm">
                                  <span className="text-green-700">{task}:</span>
                                  <span className="font-medium text-green-900">{count}</span>
                                </div>
                              ))}
                            </div>
                          )}
                          
                          {results.length_stats && (
                            <div className="p-4 bg-purple-50 rounded-lg border border-purple-200">
                              <h4 className="font-medium text-purple-900 mb-2">Length Statistics</h4>
                              <div className="text-sm space-y-1">
                                <div className="flex justify-between">
                                  <span className="text-purple-700">Hyp Avg:</span>
                                  <span className="font-medium text-purple-900">
                                    {results.length_stats.hyp_avg_length?.toFixed(0)}
                                  </span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-purple-700">Tgt Avg:</span>
                                  <span className="font-medium text-purple-900">
                                    {results.length_stats.tgt_avg_length?.toFixed(0)}
                                  </span>
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                        
                        {/* Visualizations */}
                        {results.visualizations && results.visualizations.length > 0 && (
                          <div className="space-y-4">
                            <h4 className="font-medium text-gray-900">Visualizations</h4>
                            <div className="grid gap-4">
                              {/* Bar and Pie chart (first two) */}
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <img src={results.visualizations[0]} alt="Label Distribution" className="w-full h-auto rounded-lg" />
                                <img src={results.visualizations[1]} alt="Task Distribution" className="w-full h-auto rounded-lg" />
                              </div>
                              {/* Histogram and Scatter plot (last two) */}
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <img src={results.visualizations[2]} alt="Text Length Distribution" className="w-full h-auto rounded-lg" />
                                <img src={results.visualizations[3]} alt="Hyp vs Tgt Length" className="w-full h-auto rounded-lg" />
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Train Tab */}
          <TabsContent value="train" className="space-y-6">
            <Card className="bg-white/70 backdrop-blur-sm border-white/20">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="h-6 w-6 text-indigo-600" />
                  Model Training
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {/* Training Status */}
                  {trainingStatus.status !== 'idle' && (
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-gray-900">Training Progress</span>
                        <Badge variant={
                          trainingStatus.status === 'completed' ? 'default' :
                          trainingStatus.status === 'error' ? 'destructive' : 'secondary'
                        }>
                          {trainingStatus.status}
                        </Badge>
                      </div>
                      
                      <Progress value={trainingStatus.progress} className="w-full" />
                      
                      <p className="text-sm text-gray-600">{trainingStatus.message}</p>
                      
                      {trainingStatus.status === 'training' && (
                        <div className="flex items-center gap-2 text-blue-600">
                          <Loader className="h-4 w-4 animate-spin" />
                          <span className="text-sm">Training in progress...</span>
                        </div>
                      )}
                    </div>
                  )}
                  
                  {/* Training Controls */}
                  <div className="space-y-4">
                    <div className="p-4 bg-indigo-50 rounded-lg border border-indigo-200">
                      <h3 className="font-medium text-indigo-900 mb-2">Ensemble Configuration</h3>
                      <div className="grid grid-cols-2 gap-4 text-sm text-indigo-700">
                        <div>Number of Models: <span className="font-medium">5</span></div>
                        <div>Test Split: <span className="font-medium">20%</span></div>
                        <div>Bootstrap Sampling: <span className="font-medium">Enabled</span></div>
                        <div>Early Stopping: <span className="font-medium">3 Patience</span></div>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-4">
                      <Button
                        onClick={startTraining}
                        disabled={trainingStatus.status === 'training' || trainingStatus.status === 'processing'}
                        className="flex items-center gap-2"
                      >
                        <Play className="h-4 w-4" />
                        Start Training
                      </Button>
                      
                      {modelStats.model_ready && (
                        <Alert className="flex-1">
                          <CheckCircle className="h-4 w-4" />
                          <AlertDescription>
                            Ensemble model is ready with {modelStats.models_trained} trained models.
                          </AlertDescription>
                        </Alert>
                      )}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Predict Tab */}
          <TabsContent value="predict" className="space-y-6">
            <Card className="bg-white/70 backdrop-blur-sm border-white/20">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-6 w-6 text-indigo-600" />
                  Single Prediction
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-700">Task Type</label>
                      <Select value={predictionForm.task} onValueChange={(value) => 
                        setPredictionForm(prev => ({ ...prev, task: value }))
                      }>
                        <SelectTrigger>
                          <SelectValue placeholder="Select task type" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="DM">Definition Modeling (DM)</SelectItem>
                          <SelectItem value="MT">Machine Translation (MT)</SelectItem>
                          <SelectItem value="PG">Paraphrase Generation (PG)</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-700">Hypothesis</label>
                    <Textarea
                      placeholder="Enter the generated hypothesis text..."
                      value={predictionForm.hyp}
                      onChange={(e) => setPredictionForm(prev => ({ ...prev, hyp: e.target.value }))}
                      rows={3}
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-700">Target</label>
                    <Textarea
                      placeholder="Enter the target/reference text..."
                      value={predictionForm.tgt}
                      onChange={(e) => setPredictionForm(prev => ({ ...prev, tgt: e.target.value }))}
                      rows={3}
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-700">Source</label>
                    <Textarea
                      placeholder="Enter the source text..."
                      value={predictionForm.src}
                      onChange={(e) => setPredictionForm(prev => ({ ...prev, src: e.target.value }))}
                      rows={3}
                    />
                  </div>
                  
                  <Button
                    onClick={handlePrediction}
                    disabled={!modelStats.model_ready}
                    className="w-full flex items-center justify-center gap-2"
                  >
                    <Brain className="h-4 w-4" />
                    Predict Hallucination
                  </Button>
                  
                  {/* Prediction Result */}
                  {predictionResult && (
                    <div className="mt-6 p-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-200">
                      <h3 className="font-semibold text-blue-900 mb-4">Prediction Result</h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="p-4 bg-white rounded-lg border border-blue-200">
                          <h4 className="font-medium text-blue-700 mb-2">Hallucination Type</h4>
                          <p className="text-lg font-semibold text-blue-900">
                            {predictionResult.predicted_type}
                          </p>
                        </div>
                        <div className="p-4 bg-white rounded-lg border border-blue-200">
                          <h4 className="font-medium text-blue-700 mb-2">Severity Score</h4>
                          <div className="flex items-center gap-3">
                            <Progress value={predictionResult.severity_percent} className="flex-1" />
                            <span className="font-semibold text-blue-900">
                              {predictionResult.severity_percent}%
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Batch Tab */}
          <TabsContent value="batch" className="space-y-6">
            <Card className="bg-white/70 backdrop-blur-sm border-white/20">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="h-6 w-6 text-indigo-600" />
                  Batch Processing
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {datasets.length === 0 ? (
                    <div className="text-center py-8">
                      <FileText className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                      <p className="text-gray-500">No datasets available. Upload data to get started.</p>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <p className="text-gray-600">
                        Process entire datasets for hallucination detection. Results will be available for download.
                      </p>
                      
                      {datasets.map((dataset) => (
                        <div key={dataset.id} className="p-4 bg-white rounded-lg border border-gray-200 shadow-sm">
                          <div className="flex items-center justify-between">
                            <div>
                              <h4 className="font-medium text-gray-900">{dataset.filename}</h4>
                              <p className="text-sm text-gray-500">
                                {dataset.rows} rows ready for processing
                              </p>
                            </div>
                            <div className="flex gap-2">
                              <Button
                                onClick={() => handleBatchPredict(dataset.id)}
                                disabled={!modelStats.model_ready}
                                variant="outline"
                                size="sm"
                                className="flex items-center gap-1"
                              >
                                <TrendingUp className="h-4 w-4" />
                                Process
                              </Button>
                              <Button
                                onClick={() => downloadResults(dataset.id)}
                                variant="outline"
                                size="sm"
                                className="flex items-center gap-1"
                              >
                                <Download className="h-4 w-4" />
                                Download
                              </Button>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

export default App;
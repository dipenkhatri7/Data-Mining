import argparse
import logging
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from . import config
from .data.data_loader import DataLoader
from .models.model_factory import ModelFactory
from .models.evaluation import ModelEvaluator
from .visualization.plotting import Plotter

logger = logging.getLogger(__name__)

def run_notebooks():
    """Execute Jupyter notebooks programmatically."""
    notebook_dir = Path('notebooks')
    for notebook_path in notebook_dir.glob('*.ipynb'):
        logger.info(f"Executing notebook: {notebook_path}")
        
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)
        
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': 'notebooks/'}})
        
        # Save executed notebook
        output_path = notebook_path.parent / f"executed_{notebook_path.name}"
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

def main(dataset_type: str, model_type: str):
    """Main execution function."""
    # Initialize components
    print("Initializing components...")
    data_loader = DataLoader()
    print("DataLoader initialized")
    model_factory = ModelFactory()
    print("ModelFactory initialized")
    evaluator = ModelEvaluator()
    print("ModelEvaluator initialized")
    plotter = Plotter()
    print("Plotter initialized")
    
    # Load and prepare data
    X_train, X_test, y_train, y_test = data_loader.load_and_prepare_data(dataset_type)
    print('X_train', X_train)
    print('X_test', X_test)
    print('y_train', y_train)
    print('y_test', y_test)
    # Train model
    model = model_factory.create_model(model_type)
    print("model", model)
    model.fit(X_train, y_train)
    print("Model trained")
    
    # Evaluate model
    metrics = evaluator.evaluate(model, X_test, y_test)
    logger.info(f"Model performance metrics: {metrics}")
    print(f"Model performance metrics: {metrics}")
    # Generate plots
    plotter.plot_confusion_matrix(model, X_test, y_test)
    plotter.plot_roc_curve(model, X_test, y_test)
    
    # Save model and results
    model_factory.save_model(model, f"{dataset_type}_{model_type}_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run heart disease and diabetes prediction analysis')
    parser.add_argument('--dataset', choices=['heart', 'diabetes'], required=True,
                      help='Dataset to use for analysis')
    parser.add_argument('--model', choices=list(config.MODEL_CONFIGS.keys()), required=True,
                      help='Model type to use for prediction')
    parser.add_argument('--run-notebooks', action='store_true',
                      help='Execute Jupyter notebooks programmatically')
    
    args = parser.parse_args()

    print('args', args)
    if args.run_notebooks:
        run_notebooks()
    
    main(args.dataset, args.model)
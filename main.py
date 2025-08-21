import os
from train_agent import train, evaluate_model

def main():
    model_path = 'ppo_pacman.zip'
    
    # Check if model exists
    if os.path.exists(model_path):
        print(f"Found existing trained model: {model_path}")
        print("Skipping training and proceeding to evaluation...")
    else:
        print(f"No trained model found at: {model_path}")
        print("Starting training process...")
        
        # Run training
        try:
            model = train(model_path=model_path)
            print("Training completed successfully!")
        except Exception as e:
            print(f"Training failed with error: {e}")
            return
    
    # Run evaluation
    print("\n" + "="*60)
    print("EVALUATING TRAINED AGENT")
    print("="*60)
    
    try:
        eval_results = evaluate_model(model_path=model_path, episodes=100)
        if eval_results is not None:
            print("Evaluation completed successfully!")
        else:
            print("Evaluation failed.")
            return
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        return
    
    print("\nComplete! Training and evaluation finished.")

if __name__ == "__main__":
    main()

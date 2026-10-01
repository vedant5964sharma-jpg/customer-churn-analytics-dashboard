from model import load_data, train_model, save_artifacts

if __name__ == "__main__":
    df = load_data()
    artifacts = train_model(df)
    save_artifacts(artifacts)
    print("Training complete")
    print(artifacts.metrics)

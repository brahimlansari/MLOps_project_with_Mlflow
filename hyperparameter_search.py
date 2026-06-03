from mlProject.config.configuration import ConfigurationManager
from mlProject.components.hyperparameter_tuner import HyperparameterTuner


if __name__ == "__main__":
    config = ConfigurationManager()
    search_config = config.get_model_search_config()
    tuner = HyperparameterTuner(config=search_config)
    best = tuner.run()

    print("Hyperparameter search terminé")
    if best is not None:
        print("Meilleur modèle trouvé :")
        print(f"  alpha = {best['alpha']}")
        print(f"  l1_ratio = {best['l1_ratio']}")
        print(f"  rmse = {best['rmse']}")
        print(f"  mae = {best['mae']}")
        print(f"  r2 = {best['r2']}")
        print(f"  best_model_path = {best['best_model_path']}")
    else:
        print("Aucun résultat de recherche n'a été trouvé.")

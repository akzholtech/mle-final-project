import pickle
import pandas as pd
import scipy.sparse as sp
from mlflow.pyfunc import PythonModel


class AlsModel(PythonModel):

    def __init__(self, model):
        super().__init__()
        self._model = model

    def load_context(self, context):

        with open(context.artifacts["als_model"], "rb") as f:
            self._model = pickle.load(f)

        with open(context.artifacts["user_encoder"], "rb") as f:
            self.user_encoder = pickle.load(f)
        
        with open(context.artifacts["item_encoder"], "rb") as f:
            self.item_encoder = pickle.load(f)
        
        self.user_items = sp.load_npz(context.artifacts["user_items_matrix"])
        self.popular = pd.read_parquet(context.artifacts["popular_items"])



    def predict(self, context, model_input: pd.DataFrame):
        user_id_enc = self.user_encoder.transform(model_input["user_id"])[0]
        recommendations = self._model.recommend(user_id_enc,
                                        self.user_item_matrix[user_id_enc],
                                        filter_already_liked_items=model_input["include_seen"],
                                        N=model_input["top_k"])
        recommendations = pd.DataFrame({"item_id_enc": recommendations[0], "score": recommendations[1]})
        recommendations['item_id'] = self.item_encoder.inverse_transform(recommendations["item_id_enc"])

        return recommendations
    
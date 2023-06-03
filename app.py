from flask import Flask, request
from gensim.models.doc2vec import Doc2Vec
import pandas as pd
import json
import numpy as np
from sklearn.metrics import mean_squared_error
#pip install flask
#pip install pandas
#pip install gensim
#pip install openpyxl

app = Flask(__name__)

# 전역 변수로 모델 초기화
model = None
car_data = None

def load_model():
    global model, car_data
    model = Doc2Vec.load("car_data_model_통합.d2v")
    try:
        car_data = pd.read_excel('CarData_통합ver.xlsx')
    except FileNotFoundError:
        print("CarData_통합ver.xlsx file not found.")
    except Exception as e:
        print("Error occurred while loading CarData_통합ver.xlsx:", str(e))

@app.before_request
def before_request():
    if model is None:
        load_model()


@app.route('/get_similar_cars', methods=['GET'])
def get_similar_cars():
    # 모델 사용 코드
    sentence = request.args.get('sentence')  # URL 매개변수로부터 sentence 값을 받아옴
    print(sentence)
    # sentence = "국산 RV 기아 셀토스 22000000 2020 1000 가솔린"
    new_vector = model.infer_vector(sentence.lower().split())
    similar_docs = model.dv.most_similar([new_vector], topn=10)
    
    # 결과 반환
    similar_cars = []
    for doc_tag, similarity in similar_docs:
        doc_tag = int(doc_tag)  # doc_tag를 정수로 변환
        car_info = car_data.iloc[doc_tag]
        car_info_dict = car_info.to_dict()
        car_info_dict = {str(key): value for key, value in car_info_dict.items()}  # 딕셔너리 키를 문자열로 변환
        similar_cars.append({
            # 'doc_tag': str(doc_tag),
            # 'similarity': similarity,
            'car_info': car_info_dict
        })

    return json.dumps(similar_cars, ensure_ascii=False)  # JSON 직렬화를 직접 수행하여 Response로 반환


@app.route('/personalReco', methods=['POST'])
def personalReco():
    rank_data = request.json  # JSON 형태로 전달된 데이터를 받음
    print("rank_data",rank_data)
    # 전달받은 데이터 활용
    process_received_rank_data(rank_data['data'])
    R = preprocess_data(rank_data['data'])

    # 협업 필터링 수행
    pred_matrix = perform_collaborative_filtering(R)
    print("pred_matrix: ",pred_matrix)
    print('실제 행렬:\n', R)
    print('\n예측 행렬:\n', np.round(pred_matrix, 3))

    # rank_data['id']에 해당하는 차량의 예측값 가져오기
    target_id = rank_data['id']
    print("target_id-----",target_id)
    id_to_idx = {}
    idx = 0
    for car in rank_data['data']:
        car_id = car['id']
        if car_id not in id_to_idx:
            id_to_idx[car_id] = idx
            idx += 1

    print("id-----",id_to_idx)
    if target_id in id_to_idx: #target_id가 id_to_idx 사전의 키로 존재하는지 확인
        idx = id_to_idx[target_id]
        print("idx----",idx)
        print("pred_matrix.shape[0]----",pred_matrix.shape[0])
        #idx가 pred_matrix 행의 범위 내에 있는지 확인(행의 개수는 차량 데이터 개수와 동일)
        if idx < pred_matrix.shape[0]:
            predicted_values = pred_matrix[idx]
            print(f"id가 {target_id}인 차량의 예측값:", predicted_values)
            
            id_list, cdno_list = prepreprocess_data(rank_data['data'])
            predicted_data = [{'cdno': cdno, 'predicted_value': value} for cdno, value in zip(cdno_list, predicted_values)]
            print(predicted_data)
            sorted_predicted_data = sorted(predicted_data, key=lambda x: x['predicted_value'], reverse=True)
            print(sorted_predicted_data)
            top_10_predicted_data = sorted_predicted_data[:min(10, len(sorted_predicted_data))]
            return top_10_predicted_data

        else:
            print(f"id {target_id}에 해당하는 차량의 인덱스가 유효하지 않습니다.")
            return []
    else:
        print(f"id {target_id}에 해당하는 차량이 데이터에 없습니다.")
        return []

    

def process_received_rank_data(rank_data):
    # 전달받은 데이터 처리 로직 작성
    # 예시에서는 데이터를 출력하는 것으로 대체
    for car in rank_data:
        print(car)

    return car_data

def prepreprocess_data(rank_data):
    # 데이터를 원하는 형태로 가공
    id_set = set()
    cdno_set = set()
    for data in rank_data:
        id_set.add(data['id'])
        cdno_set.add(data['cdno'])

    id_list = list(id_set)
    cdno_list = list(cdno_set)
    return id_list, cdno_list

def preprocess_data(rank_data):
    id_list, cdno_list = prepreprocess_data(rank_data)
    id_to_idx = {id_val: idx for idx, id_val in enumerate(id_list)}
    cdno_to_idx = {cdno_val: idx for idx, cdno_val in enumerate(cdno_list)}

    R = np.zeros((len(id_list), len(cdno_list)))
    for data in rank_data:
        id_idx = id_to_idx[data['id']]
        cdno_idx = cdno_to_idx[data['cdno']]
        R[id_idx, cdno_idx] = data['count']

    return R

def perform_collaborative_filtering(R):
    num_users, num_items = R.shape
    K = 3

    np.random.seed(1)
    P = np.random.normal(scale=1. / K, size=(num_users, K))
    Q = np.random.normal(scale=1. / K, size=(num_items, K))

    non_zeros = [(i, j, R[i, j]) for i in range(num_users) for j in range(num_items) if R[i, j] > 0]

    steps = 1000
    learning_rate = 0.01
    r_lambda = 0.01

    for step in range(steps):
        for i, j, r in non_zeros:
            eij = r - np.dot(P[i, :], Q[j, :].T)
            P[i, :] = P[i, :] + learning_rate * (eij * Q[j, :] - r_lambda * P[i, :])
            Q[j, :] = Q[j, :] + learning_rate * (eij * P[i, :] - r_lambda * Q[j, :])

        rmse = get_rmse(R, P, Q, non_zeros)
        if (step % 50) == 0:
            print("### iteration step : ", step, " rmse : ", rmse)

    pred_matrix = np.dot(P, Q.T)
    return pred_matrix

def get_rmse(R, P, Q, non_zeros):
    error = 0
    full_pred_matrix = np.dot(P, Q.T)
    x_non_zero_ind = [non_zero[0] for non_zero in non_zeros]
    y_non_zero_ind = [non_zero[1] for non_zero in non_zeros]
    R_non_zeros = R[x_non_zero_ind, y_non_zero_ind]
    full_pred_matrix_non_zeros = full_pred_matrix[x_non_zero_ind, y_non_zero_ind]
    mse = mean_squared_error(R_non_zeros, full_pred_matrix_non_zeros)
    rmse = np.sqrt(mse)
    return rmse


if __name__ == '__main__':
    app.run(port=3030)
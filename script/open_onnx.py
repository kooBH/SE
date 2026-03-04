import onnx
from onnx_opcounter import calculate_macs

onnx_model_path = 'mpANC.onnx' 

try:
    model = onnx.load(onnx_model_path)
    onnx.checker.check_model(model)

    print("ONNX Model Structure:")
    print("=====================")
    print(onnx.helper.printable_graph(model.graph))

    macs = calculate_macs(model)
    macs *= 125

    print("MACS:", macs)

except FileNotFoundError:
    print(f"오류: '{onnx_model_path}' 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"모델을 로드하는 중 오류가 발생했습니다: {e}")
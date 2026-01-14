import keras
import tensorflow as tf

INP = "pretrained_model.hdf5"
OUT = "pretrained_model_portable.keras"

# Cargar permitiendo Lambda (solo porque confías en el archivo)
model = keras.models.load_model(INP, compile=False, safe_mode=False)

# Mapear tensores intermedios: tensor_original -> tensor_nuevo
tensor_map = {}

# Crear nuevo input
new_inputs = keras.Input(shape=model.inputs[0].shape[1:], name=model.inputs[0].name.split(":")[0])
tensor_map[model.inputs[0]] = new_inputs

# Recorrer capas en orden y recrear el grafo
for layer in model.layers:
    if isinstance(layer, keras.layers.InputLayer):
        continue

    # Tensores de entrada a esta capa (1 o varios)
    old_inputs = layer.input if isinstance(layer.input, (list, tuple)) else [layer.input]
    new_layer_inputs = [tensor_map[t] for t in old_inputs]

    # Reemplazo EXACTO: Lambda (x/255) -> Rescaling(1/255)
    if isinstance(layer, keras.layers.Lambda) and layer.name == "lambda":
        x = new_layer_inputs[0]
        y = tf.keras.layers.Rescaling(1.0 / 255.0, name="lambda")(x)
    else:
        # Reusar la MISMA instancia de capa (mantiene config); solo la aplicamos al nuevo tensor
        y = layer(new_layer_inputs[0] if len(new_layer_inputs) == 1 else new_layer_inputs)

    # Guardar salida(s) en el mapa
    if isinstance(y, (list, tuple)):
        old_outs = layer.output
        old_outs = old_outs if isinstance(old_outs, (list, tuple)) else [old_outs]
        for ot, ny in zip(old_outs, y):
            tensor_map[ot] = ny
    else:
        tensor_map[layer.output] = y

# Salidas del modelo original
old_outputs = model.outputs if isinstance(model.outputs, (list, tuple)) else [model.outputs]
new_outputs = [tensor_map[t] for t in old_outputs]

new_model = keras.Model(inputs=new_inputs, outputs=new_outputs, name=model.name)

# Copiar pesos (Lambda/Rescaling no tienen pesos, el resto sí)
new_model.set_weights(model.get_weights())

# Guardar portable
new_model.save(OUT)
print("OK ->", OUT)


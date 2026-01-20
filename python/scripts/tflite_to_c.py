import os
import argparse

def convert_tflite_to_c(input_path, output_name):
    if not os.path.exists(input_path):
        print(f"Error: No se encontró el archivo {input_path}")
        return

    # Leer el archivo binario
    with open(input_path, 'rb') as f:
        data = f.read()

    array_name = output_name.replace(".", "_").replace("-", "_")
    
    # Generar el archivo .h
    with open(f"{output_name}.h", 'w') as h_file:
        h_file.write("#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n")
        h_file.write(f"extern const unsigned char {array_name}[];\n")
        h_file.write(f"extern const int {array_name}_len;\n\n")
        h_file.write("#endif\n")

    # Generar el archivo .cc
    with open(f"{output_name}.cc", 'w') as cc_file:
        cc_file.write(f'#include "{output_name}.h"\n\n')
        # Alineación vital para microcontroladores (M33/RP2350)
        cc_file.write(f"alignas(16) const unsigned char {array_name}[] = {{\n")
        
        # Escribir los bytes en formato hexadecimal (12 por línea)
        for i, byte in enumerate(data):
            cc_file.write(f"0x{byte:02x}, ")
            if (i + 1) % 12 == 0:
                cc_file.write("\n  ")
        
        cc_file.write("\n};\n\n")
        cc_file.write(f"const int {array_name}_len = {len(data)};\n")

    print(f"Conversión completa: {output_name}.h y {output_name}.cc generados.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convierte .tflite a arreglos de C++')
    parser.add_argument('input', help='Ruta al archivo .tflite')
    parser.add_argument('output', help='Nombre base para los archivos de salida')
    args = parser.parse_args()
    
    convert_tflite_to_c(args.input, args.output)
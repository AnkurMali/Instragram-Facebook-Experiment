import cv2
from jinja2 import FileSystemLoader, Environment
import json
import numpy
import codecs
import os
import pprint
import shutil
import time
import microsoft
import clarifai_



SETTINGS = None
def settings(name):
    
    global SETTINGS 
    if SETTINGS is None:
        SETTINGS = {
            'api_keys_filepath' : './api_keys.json',
            'input_images_dir' : 'input_images',
            'output_dir' : 'Result',
            'htmlfiles' : 'webpage',
            'output_image_height' : 200,
            'vendors' : {
                'msft' : microsoft,
                'clarifai' : clarifai_
            }
        }
        with open(SETTINGS['api_keys_filepath']) as data_file: 
            SETTINGS['api_keys'] = json.load(data_file)

    return SETTINGS[name]
        

def log_status(filepath, vendor_name, msg):
    filename = os.path.basename(filepath).encode('utf-8')
    print("%s -> %s" % ((filename + ", " + vendor_name).ljust(40), msg))


def resize_and_save(input_image_filepath, output_image_filepath):
    image = cv2.imread(input_image_filepath)
    height = image.shape[0]
    width = image.shape[1]
    aspect_ratio = float(width) / float(height)

    new_height = settings('output_image_height')
    new_width = int(aspect_ratio * new_height)

    output_image = cv2.resize(image, (new_width, new_height))
    cv2.imwrite(output_image_filepath, output_image)
    

def render_from_template(directory, template_name, **kwargs):
    loader = FileSystemLoader(directory)
    env = Environment(loader=loader)
    template = env.get_template(template_name)
    return template.render(**kwargs)


def process_all_images():

    image_results = []
    if not os.path.exists(settings('output_dir')):
        os.makedirs(settings('output_dir'))
    for filename in os.listdir(settings('input_images_dir')):
        if not filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            continue
        filepath = os.path.join(settings('input_images_dir'), filename).encode('utf-8')

        image_result = {
            'input_image_filepath' : filepath,
            'output_image_filepath' : filename,
            'vendors' : []
        }
        image_results.append(image_result)
        for vendor_name, vendor_module in sorted(settings('vendors').iteritems(), reverse=True):
            output_json_filename = filename + "." + vendor_name + ".json"
            output_json_filename = output_json_filename.encode('utf-8')
            output_json_path = os.path.join(settings('output_dir'), output_json_filename).encode('utf-8')
            output_image_filepath = os.path.join(settings('output_dir'), filename).encode('utf-8')

            if os.path.isfile(output_json_path):

                log_status(filepath, vendor_name, "skipping API call, already cached")
                with codecs.open(output_json_path, 'r',encoding='utf-8') as infile:
                    raw_api_result = infile.read()

            else:
                log_status(filepath, vendor_name, "calling API")
                raw_api_result = vendor_module.call_vision_api(filepath, settings('api_keys'))

                log_status(filepath, vendor_name, "success, storing result in %s" % output_json_path)
                with codecs.open(output_json_path, 'w',encoding='utf-8') as outfile:
                    outfile.write(raw_api_result)

                log_status(filepath, vendor_name, "writing output image in %s" % output_image_filepath)
                resize_and_save(filepath, output_image_filepath)

                time.sleep(1)

            api_result = json.loads(raw_api_result)

            standardized_result = vendor_module.get_standardized_result(api_result)
            image_result['vendors'].append({
                'api_result' : api_result,
                'vendor_name' : vendor_name,
                'standardized_result' : standardized_result,
                'output_json_filename' : output_json_filename
            })


    result_html = render_from_template('.', os.path.join(settings('htmlfiles'), 'template.html'), image_results=image_results).encode('utf-8')
    
    result_html_filepath = os.path.join(settings('output_dir'), 'result.html').encode('utf-8')
    with open(result_html_filepath, 'w') as result_html_file:
        result_html_file.write(result_html)

       
if __name__ == "__main__":
    process_all_images()


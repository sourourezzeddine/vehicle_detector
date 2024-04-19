"""! @brief module responsible for collecting all the vehicle's features into one list."""
import numpy as np

class_names = ['car','truck','LP','Toyota','Volkswagen','Ford','Honda','Chevrolet','Nissan','BMW','Mercedes','Audi','Tesla','Hyundai','Kia','Mazda','Fiat','Jeep','Porsche','Volvo','Land Rover','Peugeot','Renault','Citroen','Isuzu','MAN','Iveco','Mitsubishi','Opel','Scoda','Mini','Ferrari','Lamborghini','Jaguar','Suzuki', 'Ibiza', 'Haval','GMC']

def filter_process_objects(results):
    """Filters and processes detected objects.

    @param the results of the yolo general features model.

    @return detected objects stored in a list in the following order ['car'/'truck', 'LP', 'brand'].
    """
    for result in results:
        class_ids=[]
        confidences=[]
        for result in results:
            boxes = result.boxes.cpu().numpy()
            confidences.append(boxes.conf)
            class_ids.append(boxes.cls)
            class_ids_array = np.concatenate(class_ids)
            confidences_array = np.concatenate(confidences)
            desired_class_ids_array = class_ids_array.astype(int).tolist()
            desired_conf_array = confidences_array.astype(float).tolist()
            result_dict = {}
            car_found = False
            LP_found = False
            Brand_found = False

            for i in range(len(desired_class_ids_array)):
                if desired_conf_array[i] > 0.3:
                    result_dict[desired_class_ids_array[i]] = desired_conf_array[i]

            result_dict = dict(sorted(result_dict.items()))

            if len(result_dict)>3: 
                class_ids_list = list(result_dict.keys())
                if class_ids_list[0]=='0' and class_ids_list[1]=='1': # if the model is unable to decide whether it's a car or a truck, the one with the higher confidence is gonna be kept
                    if result_dict['0']> result_dict['1']:
                        del result_dict['1']
                    else:
                        del result_dict['0']
                class_ids_list1 = list(result_dict.keys())
                while len(result_dict)>3:  # choose the brand with the higher confidence
                    if result_dict[class_ids_list1[-1]]>result_dict[class_ids_list1[-2]]:
                        del result_dict[class_ids_list1[-2]]
                    else:
                        del result_dict[class_ids_list1[-1]]
                    class_ids_list1 = list(result_dict.keys())

            features = list(result_dict.keys())
            final_features = []
            if len(result_dict) > 2:
                for j in result_dict.keys():
                    if j in [0, 1] and result_dict[j] > 0.1:  
                        car_found = True
                    elif j == 2 and result_dict[j] > 0.5:
                        LP_found = True
                    elif 3 <= j <= 35 and result_dict[j] > 0.5:
                        Brand_found = True
                if car_found and LP_found and Brand_found: # if the vehicle and the brand and the lp are detected proceed to other tests
                    for feature in features:
                        final_features.append(class_names[feature])
                    return final_features
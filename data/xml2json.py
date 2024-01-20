# pip install lxml
import os
import json
import xml.etree.ElementTree as ET

START_BOUNDING_BOX_ID = 1


def get(root, name):
    return root.findall(name)


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def convert(xml_dir, xml_list, json_file):
    json_dict = {"images": [], "annotations": [], "categories": []}
    categories = pre_define_categories.copy()
    bnd_id = START_BOUNDING_BOX_ID
    all_categories = {}
    for index, line in enumerate(xml_list):
        # if index == 5:
        #     break
        # print("Processing %s"%(line))
        xml_f = os.path.join(xml_dir, line + '.xml')
        tree = ET.parse(xml_f)
        root = tree.getroot()

        filename = os.path.basename(xml_f)[:-4] + ".bmp"
        # filename = os.path.basename(xml_f)[:-4] + ".jpg"

        image_id = index

        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename, 'height': height, 'width': width, 'id': image_id}
        json_dict['images'].append(image)

        for obj in root.iter('object'):
            category = get_and_check(obj, 'name', 1).text
            if category in all_categories:
                all_categories[category] += 1
            else:
                all_categories[category] = 1
            if category not in categories:
                print(category)
                if only_care_pre_define_categories:
                    continue
                new_id = len(categories) + 1
                print(
                    "[warning] category '{}' not in 'pre_define_categories'({}), create new id: {} automatically".format(
                        category, pre_define_categories, new_id))
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(float(get_and_check(bndbox, 'xmin', 1).text))
            ymin = int(float(get_and_check(bndbox, 'ymin', 1).text))
            xmax = int(float(get_and_check(bndbox, 'xmax', 1).text))
            ymax = int(float(get_and_check(bndbox, 'ymax', 1).text))
            assert (xmax > xmin), "xmax <= xmin, {}".format(line)
            assert (ymax > ymin), "ymax <= ymin, {}".format(line)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            x = xmin
            y = ymin
            ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id':
                image_id, 'bbox': [x, y, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0}

            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)

    with open(json_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(json_dict, indent=4, ensure_ascii=False))

    print("------------create {} done--------------".format(json_file))
    print("find {} categories: {} -->>> your pre_define_categories {}: {}".format(len(all_categories),
                                                                                  all_categories.keys(),
                                                                                  len(pre_define_categories),
                                                                                  pre_define_categories.keys()))
    print("category: id --> {}".format(categories))
    print(categories.keys())
    print(categories.values())


if __name__ == '__main__':
    classes = ['ball', 'cylinder', 'square cage', 'cube',
               'circle cage', 'human body', 'metal bucket', 'tyre']
    pre_define_categories = {}
    for i, cls in enumerate(classes):
        pre_define_categories[cls] = i + 1

    only_care_pre_define_categories = True

    OS = 'obj'
    sets = ['train', 'test']
    xml_dir = '/'
    save_path = '/'
    main_dir = os.path.join('./VOC', 'ImageSets/Main')

    print('xml_dir is {}'.format(xml_dir))
    for st in sets:
        save_json = os.path.join(save_path, '%s2023.json' % st)
        with open(os.path.join(main_dir, '%s.txt' % st), 'r') as f:
            lines = f.readlines()
        list = [x.strip() for x in lines]
        convert(xml_dir, list, save_json)

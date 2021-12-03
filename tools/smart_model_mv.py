import os
import shutil

if __name__ == '__main__':
    root = '/raid/results/label_optimization/cifar100/vanilla'
    dest = '/raid/results/label_optimization/cifar100/smart_models'

    set_model = dict()
    for path, dir, files in os.walk(root):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.json':
#                 print('%s%s' % (path, filename))
# /raid/results/label_optimization/cifar100/vanilla/resnet110/202103250544/model_ckp
                split_path = path.split('/')
                model = split_path[-2]
                date = split_path[-1]
                if model not in set_model:
                    set_model[model] = 1
                
                model_path = os.path.join(path, 'model_ckp', 'student_lastest.pth')
                dest_path = os.path.join(dest, model, str(set_model[model]))
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)

                shutil.copy(model_path, dest_path)
                set_model[model] += 1

                
                
                

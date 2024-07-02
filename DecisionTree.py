"""
    Assume input tensor is of the form:
    tensor = [outlook,temp,humidity,windy,play]
    here play is the target variable (class)
    remaining four are explanatory variables

"""
import torch
import math

"""Calculate the entropy of the entire dataset"""
# input:tensor
# output:int/float
def get_entropy_of_dataset(tensor:torch.Tensor):
    c,yes_no = torch.unique(tensor[:,-1],return_counts = True) # target column
    pos_neg = yes_no/tensor.shape[0] #14
    probability = pos_neg.tolist()
    entropy = 0.0
    num1 = probability[0]
    num2 = probability[1]
    if num1 == 0.0:
        if num2 == 0.0:
            entropy = 0.0
        else:
            entropy += -(num2 * math.log2(num2))

    elif num2 == 0.0:
        entropy += -(num1 * math.log2(num1))

    else:
        entropy = -(num1 * math.log2(num1)) -(num2 * math.log2(num2))
    return entropy


"""Return avg_info of the attribute provided as parameter"""
# input:tensor,attribute number 
# output:int/float
def get_avg_info_of_attribute(tensor:torch.Tensor, attribute:int):
    c,freq = torch.unique(tensor[:,attribute],return_counts = True)
    variations = c.tolist()
    avg_info = 0.0
    for j in range(len(variations)):
        pos = 0
        neg = 0
        for k in tensor:
            if k[attribute] == variations[j]:
                if k[4] == 1:
                    pos += 1
                else:
                    neg += 1
        value_pos = pos/(pos+neg)
        value_neg = neg/(pos+neg)
        if value_pos == 0.0:
            if value_neg == 0.0:
                entropy = 0.0
            else:
                entropy = -value_neg * math.log2(value_neg)
            
        elif value_neg == 0.0:
            entropy = -value_pos * math.log2(value_pos)

        else:
            entropy = -value_pos * math.log2(value_pos) + -value_neg * math.log2(value_neg)

        avg_info += ((pos+neg)/tensor.shape[0]) * entropy
    return avg_info



"""Return Information Gain of the attribute provided as parameter"""
# input:tensor,attribute number
# output:int/float
def get_information_gain(tensor:torch.Tensor, attribute:int):
    S = get_entropy_of_dataset(tensor)
    I_A = get_avg_info_of_attribute(tensor,attribute)
    information_gain = S - I_A
    return information_gain



# input: tensor
# output: ({dict},int)
def get_selected_attribute(tensor:torch.Tensor):
    """
    Return a tuple with the first element as a dictionary which has IG of all columns
    and the second element as an integer representing attribute number of selected attribute

    example : ({0: 0.123, 1: 0.768, 2: 1.23} , 2)
    """
    columns = [0,1,2,3]
    ig = list()
    maxValue = 0.0
    for i in columns:
        gain = get_information_gain(tensor,i)
        gain = round(gain,5)
        ig.append(gain)
        if maxValue < gain :
            maxValue = gain
            selected_attribute = i
    final = dict(zip(columns, ig))
    output = list()
    output.append(final)
    output.append(selected_attribute)
    return output


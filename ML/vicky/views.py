from django.shortcuts import render,redirect
from django.http import JsonResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Embedding,LSTM,SimpleRNN,GRU
from tensorflow.keras.utils import to_categorical
import numpy as np
from matplotlib import pyplot as plt
import sys
from PIL import Image
import cv2
import base64
import io
# Create your views here.

def hi(request):
        return redirect("bye/")

def bye(request):
        return render(request,"rail.html",{"v":"vicky da"})

@api_view(["POST"])
def api(x):
        try:
                output=""
                dt=json.loads(x.body)
                X=np.array(dt["X"])
                y=np.array(dt["y"])
                
                test_size=dt["test_size"]
                if dt["test_train"]==1:
                    if test_size==0:
                        if test_size>len(y) or test_size==0 or len(np.unique(y))>test_size :
                            test_size=len(np.unique(y))

                if dt["test_train"]==0:
                    x_train=X
                    y_train=y
                else:
                    x_train=X[:X.shape[0]-test_size]
                    y_train=y[:X.shape[0]-test_size]
                    x_test=X[X.shape[0]-test_size:]
                    y_test=y[X.shape[0]-test_size:]
                  

                padding=["valid","same"]
                h_act=['relu','sigmoid','tanh','softmax']
                o_act=['softmax','sigmoid','relu']
                loss=['categorical_crossentropy','mse']
                optimizer=['adam','sgd','RMSprop','Adadelta','Adagrad','Adamax','Nadam','Ftrl']
                metrics=['accuracy','AUC','Precision','Recall']
                D_act=['relu','sigmoid','softmax']
                uni_y=len(np.unique(y_train))




                if dt["normalize_data"]==0:
                    if dt["test_train"]==0:
                        x_train=x_train/x_train.max()
                    else:
                        x_train=x_train/x_train.max()
                        x_test=x_test/x_test.max()
                    if dt["pree"]:
                        pred=np.array(dt["pred"])/np.array(dt["pred"]).max()

                input_shape=x_train.shape
                model=Sequential([Conv2D(dt["num_filters"][0],dt["filter_size"][0],strides=dt["strides"][0],padding=padding[dt["pd"]],input_shape=input_shape[1:],activation=h_act[dt["ha"]],use_bias=dt["use_bias"])]+[Conv2D(dt["num_filters"][i+1],dt["filter_size"][i+1],strides=dt["strides"][i+1],padding=padding[dt["pd"]],activation=h_act[dt["ha"]],use_bias=dt["use_bias"]) for i in range(dt["n_hidden_layers"]-1)])
                model.add(MaxPooling2D(pool_size=dt["pool_size"]))
                model.add(Flatten())
                for i in range(dt["no_of_hidden_dense_layer"]):
                    model.add(Dense(dt["no_of_dense_layer_neurons"][i],activation=D_act[dt["dc"]]))
                model.add(Dense(uni_y,activation=o_act[dt["oc"]]))

                
                model.compile(optimizer=optimizer[dt["opt"]],loss=loss[dt["lss"]],metrics=[metrics[dt["mt"]]])
                model.fit(x_train,to_categorical(y_train),epochs=dt["epochs"],verbose=0)

                if dt["test_train"]==0:
                    score=model.evaluate(x_train,to_categorical(y_train),verbose=0)
                else:
                    score=model.evaluate(x_test,to_categorical(y_test),verbose=0)

                output+='Test Lose : '+str(score[0])+"\n\n"
                output+='Test accuracy : '+str(score[1])+"\n\n"
                if dt["pree"]:
                    pred=model.predict(pred)
                    output+='predictions : '+str(np.argmax(np.array(pred),axis=1))+"\n\n"

                return JsonResponse(output,safe=False)

        except ValueError as e:
                return Response(e.args[0],status.HTTP_400_BAD_REQUEST)

@api_view(["POST"])
def api2(x):
    try:
                    output=""
                    dt=json.loads(x.body)
                    test_train=dt["test_train"]
                    data=np.array(dt["data"])
                    target=np.array(dt["target"])
                    if test_train==1:
                        x_test=np.array(dt["x_test"])
                        y_test=np.array(dt["y_test"])
                        x_train=np.array(dt["x_train"])
                        y_train=np.array(dt["y_train"])
                        target=np.array(dt["target"])
                    loss=['mse','mean_absolute_error','categorical_crossentropy','binary_crossentropy']
                    optimizer=['adam','sgd','RMSprop','Adadelta','Adagrad','Adamax','Nadam','Ftrl']
                    metrics=['accuracy','AUC','Precision','Recall']
                    activation=['tanh','sigmoid','relu','softmax']
                    o_act=['relu','softmax','sigmoid']
                    D_act=['relu','sigmoid','softmax']

                    act=dt["act"]
                    use_bias=dt["use_bias"]
                    if use_bias==0:
                        use_bias=True
                    else:
                        use_bias=False
                    n_hidden_layers=dt["n_hidden_layers"]
                    ty=dt["ty"]
                    dropout=dt["dropout"]
                    no_of_units=dt["no_of_units"]
                    if ty==3:
                        time_steps=dt["time_steps"]
                        n_features=dt["n_features"]
                    if ty==2:
                        no_of_features=dt["no_of_features"]
                    add_dense_layer=dt["add_dense_layer"]
                    no_of_hidden_dense_layer=dt["no_of_hidden_dense_layer"]
                    no_of_dense_layer_neurons=dt["no_of_dense_layer_neurons"]
                    oc=dt["oc"]
                    dc=dt["dc"]
                    if ty==0 or ty==3:
                        f=dt["f"]
                    los=dt["los"]
                    mt=dt["mt"]
                    epochs=dt["epochs"]
                    ma=dt["ma"]
                    ma1=dt["ma1"]
                    if ty==3:
                        max_td=dt["max_td"]
                        predict_future=dt["predict_future"]
                        predict_next=dt["predict_next"]
                        timeseries_data=np.array(dt["timeseries_data"])

                    op=dt["op"]
                    pre=dt["pre"]
                    if pre!=[] and pre!="":
                        pre=np.array(dt["pre"])
                        try:
                            pred=dt["pred"]
                        except:
                            pred=""
                    preee=dt["preee"]
                    pree=dt["pree"]

                    uni_y=len(np.unique(target))
                    lstm=[]
                    if n_hidden_layers>1:
                        for i in range(n_hidden_layers):
                            if ty==0:
                                if i!=0:
                                    if i!=n_hidden_layers-1:
                                        lstm.append(GRU(1,activation=activation[act],use_bias=use_bias,return_sequences=True, dropout=dropout))
                                    else:
                                        lstm.append(GRU(1,activation=activation[act],use_bias=use_bias,return_sequences=False, dropout=dropout))
                                else:
                                    lstm.append(GRU(1, batch_input_shape=(None,None,data.shape[2]),activation=activation[act],use_bias=use_bias,return_sequences=True, dropout=dropout))
                            elif ty==1:
                                if i!=0:
                                    if i!=n_hidden_layers-1:
                                        lstm.append(GRU(no_of_units[i],activation=activation[act],use_bias=use_bias,return_sequences=True, dropout=dropout))
                                    else:
                                        lstm.append(GRU(no_of_units[i],activation=activation[act],use_bias=use_bias,return_sequences=False, dropout=dropout))
                                else:
                                    lstm.append(GRU(no_of_units[i], batch_input_shape=(None,None,data.shape[2]),activation=activation[act],use_bias=use_bias,return_sequences=True, dropout=dropout))
                            elif ty==2:
                                if i!=0:
                                    if i!=n_hidden_layers-1:
                                        lstm.append(GRU(no_of_units[i],activation=activation[act],use_bias=use_bias,return_sequences=True, dropout=dropout))
                                    else:
                                        lstm.append(GRU(no_of_units[i],activation=activation[act],use_bias=use_bias,return_sequences=False, dropout=dropout))
                                else:
                                    lstm.append(GRU(no_of_units[i],activation=activation[act],use_bias=use_bias,return_sequences=True, dropout=dropout))
                            elif ty==3:
                                if i!=0:
                                    if i!=n_hidden_layers-1:
                                        lstm.append(GRU(no_of_units[i],activation=activation[act],return_sequences=True,use_bias=use_bias, dropout=dropout))
                                    else:
                                        lstm.append(GRU(no_of_units[i],activation=activation[act],return_sequences=False,use_bias=use_bias, dropout=dropout))
                                else:
                                    lstm.append(GRU(no_of_units[i], input_shape=(time_steps, n_features),activation=activation[act],return_sequences=True,use_bias=use_bias, dropout=dropout))
                    else:
                        if ty==0:
                            lstm.append(GRU(1, batch_input_shape=(None,None,data.shape[2]),use_bias=use_bias,return_sequences=False, dropout=dropout))
                        elif ty==1:
                            lstm.append(GRU(no_of_units[0],batch_input_shape=(None,None,data.shape[2]),activation=activation[act],use_bias=use_bias,return_sequences=False, dropout=dropout))
                        elif ty==2:
                            lstm.append(GRU(no_of_units[0],activation=activation[act],use_bias=use_bias,return_sequences=False, dropout=dropout))
                        elif ty==3:
                            lstm.append(GRU(no_of_units[0],activation=activation[act],input_shape=(time_steps, n_features),use_bias=use_bias,return_sequences=False, dropout=dropout))

                    if ty!=2:
                        model = Sequential(lstm)
                    else:
                        model = Sequential()
                        model.add(Embedding(2000,no_of_features,input_length=data.shape[1]))
                        for i in lstm:
                            model.add(i)
                    if add_dense_layer==0:
                        if ty==0 or ty==3:
                            for i in range(no_of_hidden_dense_layer):
                                model.add(Dense(no_of_dense_layer_neurons[i],activation=D_act[dc]))
                            model.add(Dense(f,activation=o_act[oc]))
                        elif ty==1:
                            for i in range(no_of_hidden_dense_layer):
                                model.add(Dense(no_of_dense_layer_neurons[i],activation=D_act[dc]))
                            model.add(Dense(uni_y,activation=o_act[oc]))
                        elif ty==2:
                            for i in range(no_of_hidden_dense_layer):
                                model.add(Dense(no_of_dense_layer_neurons[i],activation=D_act[dc]))
                            model.add(Dense(2,activation=o_act[oc]))
                    model.compile(loss=loss[los], optimizer=optimizer[op],metrics=[metrics[mt]])   
                    if test_train==0:
                        his=model.fit(data, target, epochs=epochs, shuffle=False,verbose=0)
                    else:
                        his=model.fit(x_train, y_train, epochs=epochs, shuffle=False,verbose=0)


                    if test_train==0:
                        score=model.evaluate(data,target,verbose=0)
                        output+='Trained data Lose : '+str(score[0])+"\n\n"
                    else:
                        score=model.evaluate(x_test,y_test,verbose=0)
                        output+='Tested data Lose : '+str(score[0])+"\n\n"
                    if ty==1 or ty==2:
                            if test_train==0:
                                    output+='Trained data Accuracy : '+str(score[1])+"\n\n"
                            else:
                                    output+='Tested data Accuracy : '+str(score[1])+"\n\n"

                    if test_train==0 and ty!=3:
                        fig=plt.figure(facecolor='lightgreen')
                        ax=plt.axes()
                        ax.set_facecolor('lightgreen')
                        ay=fig.add_subplot(1,1,1)
                        plt.title("Target(green) vs predicted(Blue)")
                        if ty==0:
                            predict = model.predict(data)
                            ay.scatter(range(len(data)),predict,c='r')
                        else:
                            pret=model.predict(data)
                            predict = np.argmax(pret,axis=1)
                            b=[pret[i][predict[i]] for i in range(len(predict))]
                            ay.scatter(range(len(data)),b,c='r')
                        ay.scatter(range(len(data)),target,c='g')
                        fig.canvas.draw()
                        img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                        img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                        pil_im=Image.fromarray(img)
                        buff=io.BytesIO()
                        pil_im.save(buff,format="PNG")
                        img_str=base64.b64encode(buff.getvalue())
                        
                        if ty==0:
                            p=np.ceil(((model.predict(data).reshape(1,len(data))[0])*ma))
                            output+='Accuracy : '+str(sum([(target[i]*ma1)==p[i] for i in range(len(data))])/len(data))+"\n\n"
                        else:
                                pred=model.predict(data)
                                predict = np.argmax(pred,axis=1)
                        if(len(data)<=20):
                            if ty==0:
                                output+="Predicted "+str(sum(((target*ma1)==p)))+" Correctly Out of "+str(len(target))+"\n\n"
                                output+="predicted output value of trained data:\t"+str(p)+"\n\n"
                            elif ty==1:
                                output+="Predicted "+str(sum((target==predict)))+" Correctly Out of "+str(len(target))+"\n\n"
                                output+="predicted output value of trained data:\t"+str(predict)+"\n\n"
                            elif ty==2:
                                output+="Predicted "+str(sum((target==predict)))+" Correctly Out of "+str(len(target[:-1]))+"\n\n"
                                output+="predicted output value of trained data:\t"+str(predict[:-1])+"\n\n"
                        else:
                            if ty==0:
                                output+="Predicted "+str(sum(((target*ma1)==p)))+" Correctly Out of "+str(len(target))+"\n\n"
                                output+="predicted first 20 sample output value of trained data:\t"+str(np.ceil((model.predict(data[:20]).reshape(1,20)[0])*ma))+"\n\n"
                            elif ty==1:
                                output+="Predicted "+str(sum((target==predict)))+" Correctly Out of "+str(len(target))+"\n\n"
                                output+="predicted first 20 sample output value of trained data:\t"+str(predict[:20])+"\n\n"
                            elif ty==2:
                                output+="Predicted "+str(sum((target==predict)))+" Correctly Out of "+str(len(target[:-1]))+"\n\n"
                                output+="predicted output value of trained data:\t"+str(predict[:-1])+"\n\n"
                    

                    elif ty!=3:
                        fig=plt.figure(facecolor='lightgreen')
                        ax=plt.axes()
                        ax.set_facecolor('lightgreen')
                        ay=fig.add_subplot(1,1,1)
                        if ty==0:
                            predict = model.predict(x_test)
                            ay.scatter(range(len(x_test)),predict,c='r')
                        else:
                            pret=model.predict(x_test)
                            predict = np.argmax(pret,axis=1)
                            b=[pret[i][predict[i]] for i in range(len(predict))]
                            ay.scatter(range(len(x_test)),b,c='r')
                        plt.title("y_test(green) vs y_predicted(Blue)")
                        ay.scatter(range(len(x_test)),y_test,c='g')
                        fig.canvas.draw()
                        img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                        img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                        pil_im=Image.fromarray(img)
                        buff=io.BytesIO()
                        pil_im.save(buff,format="PNG")
                        img_str=base64.b64encode(buff.getvalue())
                        if ty==0:
                            p=np.ceil(((model.predict(x_test).reshape(1,len(x_test))[0])*ma))
                            pa=np.ceil(((model.predict(x_train).reshape(1,len(x_train))[0])*ma))
                            output+='Accuracy : '+str(sum([(y_test[i]*ma1)==p[i] for i in range(len(x_test))])/len(x_test))+"\n\n"
                        else:
                            predict = np.argmax(model.predict(x_test),axis=1)
                            predic = np.argmax(model.predict(x_train),axis=1)
                        if(len(x_train)<=20):
                            if ty==0:
                                output+="Predicted "+str(sum(((y_test*ma1)==p)))+" Correctly Out of "+str(len(y_test))+"\n\n"
                                output+="predicted output value of trained data:\t"+str(pa)+"\n\n"
                            else:
                                output+="Predicted "+str(sum((y_test==predict)))+" Correctly Out of "+str(len(y_test))+"\n\n"
                                output+="predicted output value of trained data:\t"+str(predic)+"\n\n"
                        else:
                            if ty==0:
                                output+="Predicted "+str(sum(((y_test*ma1)==p)))+" Correctly Out of "+str(len(y_test))+"\n\n"
                                output+="predicted first 20 sample output value of trained data:\t"+str(np.ceil((model.predict(x_train[:20]).reshape(1,20)[0])*ma))+"\n\n"
                            else:
                                output+="Predicted "+str(sum((y_test==predict)))+" Correctly Out of "+str(len(y_test))+"\n\n"
                                output+="predicted first 20 sample output value of trained data:\t"+str(predic[:20])+"\n\n"
                    fig=plt.figure(facecolor='lightgreen')
                    ax=plt.axes()
                    ax.set_facecolor('lightgreen')
                    ay=fig.add_subplot(1,1,1)            
                    plt.title("Loss")
                    ay.plot(his.history['loss'])
                    fig.canvas.draw()
                    img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                    img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                    pil_im=Image.fromarray(img)
                    buff=io.BytesIO()
                    pil_im.save(buff,format="PNG")
                    img_str2=base64.b64encode(buff.getvalue())
                    
                    if pree or preee:
                        if ty==0:
                            pre1=model.predict(pre).reshape(1,len(pred))[0]
                            output+="Predicted output of user's input :"+str(np.ceil((pre1*ma)))+"\n\n"
                        elif ty==1:
                            pre1=model.predict(pre)
                            output+="Predicted output of user's input :"+str(np.argmax(pre1,axis=1))+"\n\n"
                        elif ty==2:
                            pre1=model.predict(pre)
                            output+="Predicted output of user's input :"+str(np.argmax(pre1,axis=1))+"\n\n"
                        elif ty==3:
                                try:
                                        a=model.predict(pre)
                                        if n_features==1 or f==1:
                                                trained_pre=[i[0] for i in a]
                                        else:
                                                trained_pre=a.tolist()
                                        output+="Predicted output of user's input :"+str(trained_pre)+"\n\n"
                                except:
                                        output+="Please check your input for Prediction. The time_steps number of data requirerd to predict one output and requiers all trained features to predict future."+"\n\n"

                    if ty==3:
                        target=target*ma1
                        if test_train==0:
                                a=model.predict(data)*ma1
                        else:
                                a=model.predict(x_train)*ma1
                        if n_features==1 or f==1:
                                trained_pre=[i[0] for i in a]
                        else:
                                trained_pre=a.tolist()
                        if len(data)<=20:
                                output+="Predicted output of trained data :\t"+str(trained_pre)+"\n\n"
                        else:
                                output+="Predicted first 20 sample output value of trained data :\t"+str(trained_pre[:20])+"\n\n"
                        if n_features==1 or f==1:
                                yy=[i[0][0] for i in target]
                        else:
                                yy=np.array([target[i][0] for i in range(len(target))])
                        fig=plt.figure(facecolor='lightgreen')
                        ax=plt.axes()
                        ax.set_facecolor('lightgreen')
                        if f>1:
                                for i in range(1,len(a[0])+1):
                                        plt.subplot(len(a[0])-1,2,i)
                                        plt.title("Column-"+str(i)+" series")
                                        plt.plot(yy[:,i-1:i],c='green')
                                        plt.plot(a[:,i-1:i],c='red')
                        else:
                                plt.plot(yy,c='green')
                                plt.plot(a,c='red')
                        plt.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.9)
                        plt.suptitle("Trained(green) vs Predicted(blue)")
                        fig.canvas.draw()
                        img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                        img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                        pil_im=Image.fromarray(img)
                        buff=io.BytesIO()
                        pil_im.save(buff,format="PNG")
                        img_str22=base64.b64encode(buff.getvalue())


                        if test_train==1:
                                y_test=y_test*ma1
                                a=model.predict(x_test)*ma1
                                fig=plt.figure(facecolor='lightgreen')
                                ax=plt.axes()
                                ax.set_facecolor('lightgreen')
                                if n_features==1 or f==1:
                                        trained_pre=[i[0] for i in a]
                                else:
                                        trained_pre=a.tolist()
                                if len(data)<=20:
                                        output+="Predicted output of tested data :"+str(trained_pre)+"\n\n"
                                else:
                                        output+="Predicted first 20 sample output value of tested data :\t"+str(trained_pre[:20])+"\n\n"
                                if n_features==1 or f==1:
                                        yy=[i[0][0] for i in y_test]
                                else:
                                        yy=np.array([y_test[i][0] for i in range(len(y_test))])
                                if f>1:
                                        for i in range(1,len(a[0])+1):
                                                plt.subplot(len(a[0])-1,2,i)
                                                plt.title("Column-"+str(i)+" series")
                                                plt.plot(yy[:,i-1:i],c='green')
                                                plt.plot(a[:,i-1:i],c='red')
                                else:
                                        plt.plot(yy,c='green')
                                        plt.plot(a,c='red')
                                plt.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.9)
                                plt.suptitle("y_test(green) vs y_Predicted(blue)")
                                fig.canvas.draw()
                                img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                                img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                                img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                                pil_im=Image.fromarray(img)
                                buff=io.BytesIO()
                                pil_im.save(buff,format="PNG")
                                img_str24=base64.b64encode(buff.getvalue())
                                
                        if predict_future==1:
                                x_input = np.array(timeseries_data[len(timeseries_data)-time_steps:])
                                temp_input=list(x_input)
                                lst_output=[]
                                i=0
                                while(i<predict_next):
                                    if(len(temp_input)>time_steps):
                                        x_input=np.array(temp_input[1:])
                                        x_input = x_input.reshape((1, time_steps, n_features))
                                        yhat = model.predict(x_input, verbose=0)
                                        if n_features>1:
                                                temp_input.append(yhat[0])
                                        else:
                                                temp_input.append(yhat[0][0])
                                        temp_input=temp_input[1:]
                                        if n_features>1:
                                                lst_output.append(yhat[0])
                                        else:
                                                lst_output.append(yhat[0][0])
                                        i=i+1
                                    else:
                                        x_input = x_input.reshape((1, time_steps, n_features))
                                        yhat = model.predict(x_input, verbose=0)
                                        if n_features>1:
                                                temp_input.append(yhat[0])
                                                lst_output.append(yhat[0])
                                        else:
                                                temp_input.append(yhat[0][0])
                                                lst_output.append(yhat[0][0])
                                        i=i+1

                                output+="Predicted next "+str(predict_next)+" values :"+"\n\n"

                                lst_output=np.array(lst_output)*ma1
                                output+=str(lst_output)+"\n\n"

                                timeseries_data*=max_td
                                day_new=np.arange(1,len(timeseries_data)+1)
                                day_pred=np.arange(len(timeseries_data)+1,len(timeseries_data)+predict_next+1)
                                if f>1:
                                        fig=plt.figure(facecolor='lightgreen')
                                        ax=plt.axes()
                                        ax.set_facecolor('lightgreen')
                                        for i in range(1,len(timeseries_data[0])+1):
                                                plt.subplot(len(timeseries_data[0])-1,2,i)
                                                plt.title("Column-"+str(i)+" series")
                                                plt.plot(day_new,np.array(timeseries_data)[:,i-1:i],c='green')
                                                plt.plot(day_pred,np.array(lst_output)[:,i-1:i],c='red')
                                        plt.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.9)
                                else:
                                        plt.plot(day_new,timeseries_data,c='green')
                                        plt.plot(day_pred,lst_output,c='red')
                                plt.suptitle("Trained(green) vs Predicted future "+str(predict_next)+" values(blue)")
                                fig.canvas.draw()
                                img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                                img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                                img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                                pil_im=Image.fromarray(img)
                                buff=io.BytesIO()
                                pil_im.save(buff,format="PNG")
                                img_str23=base64.b64encode(buff.getvalue())
                                
                    output+="\n"
                    if ty!=3:
                            #return output,""+str(img_str2,'utf-8'),""+str(img_str,'utf-8'),"",""
                            return JsonResponse([output,""+str(img_str2,'utf-8'),""+str(img_str,'utf-8'),"",""],safe=False)
                    else:
                            if predict_future==1 and test_train==1:
                                    #return output,""+str(img_str2,'utf-8'),""+str(img_str22,'utf-8'),""+str(img_str24,'utf-8'),""+str(img_str23,'utf-8'),""
                                    return JsonResponse([output,""+str(img_str2,'utf-8'),""+str(img_str22,'utf-8'),""+str(img_str24,'utf-8'),""+str(img_str23,'utf-8'),""],safe=False)
                            elif predict_future==0 and test_train==1:
                                #return output,""+str(img_str2,'utf-8'),""+str(img_str22,'utf-8'),""+str(img_str24,'utf-8'),"",""
                                return JsonResponse([output,""+str(img_str2,'utf-8'),""+str(img_str22,'utf-8'),""+str(img_str24,'utf-8'),"",""],safe=False)
                            elif predict_future==1 and test_train==0:
                                #return output,""+str(img_str2,'utf-8'),""+str(img_str22,'utf-8'),"",""+str(img_str23,'utf-8'),""
                                return JsonResponse([output,""+str(img_str2,'utf-8'),""+str(img_str22,'utf-8'),"",""+str(img_str23,'utf-8'),""],safe=False)
                            else:
                                    #return output,""+str(img_str2,'utf-8'),""+str(img_str22,'utf-8'),"","","" 
                                    return JsonResponse([output,""+str(img_str2,'utf-8'),""+str(img_str22,'utf-8'),"","",""],safe=False)
                                
    except Exception as e:
        #return Response(e.args[0],status.HTTP_400_BAD_REQUEST)                    
        exception_type, exception_object, exception_traceback = sys.exc_info()
        line_number = exception_traceback.tb_lineno
        output="Something went wrong... Please give correct input"
        return JsonResponse(str(line_number)+str(exception_type)+" "+str(e)+"error",safe=False)


@api_view(["POST"])
def api3(x):
    try:
                    output=""
                    dt=json.loads(x.body)
                    test_train=dt["test_train"]
                    data=np.array(dt["data"])
                    target=np.array(dt["target"])
                    if test_train==1:
                        x_test=np.array(dt["x_test"])
                        y_test=np.array(dt["y_test"])
                        x_train=np.array(dt["x_train"])
                        y_train=np.array(dt["y_train"])
                        target=np.array(dt["target"])
                    loss=['mse','mean_absolute_error','categorical_crossentropy','binary_crossentropy']
                    optimizer=['adam','sgd','RMSprop','Adadelta','Adagrad','Adamax','Nadam','Ftrl']
                    metrics=['accuracy','AUC','Precision','Recall']
                    activation=['tanh','sigmoid','relu','softmax']
                    o_act=['relu','softmax','sigmoid']
                    D_act=['relu','sigmoid','softmax']

                    act=dt["act"]
                    use_bias=dt["use_bias"]
                    if use_bias==0:
                        use_bias=True
                    else:
                        use_bias=False
                    n_hidden_layers=dt["n_hidden_layers"]
                    ty=dt["ty"]
                    dropout=dt["dropout"]
                    no_of_units=dt["no_of_units"]
                    if ty==3:
                        time_steps=dt["time_steps"]
                        n_features=dt["n_features"]
                    if ty==2:
                        no_of_features=dt["no_of_features"]
                    add_dense_layer=dt["add_dense_layer"]
                    no_of_hidden_dense_layer=dt["no_of_hidden_dense_layer"]
                    no_of_dense_layer_neurons=dt["no_of_dense_layer_neurons"]
                    oc=dt["oc"]
                    dc=dt["dc"]
                    if ty==0 or ty==3:
                        f=dt["f"]
                    los=dt["los"]
                    mt=dt["mt"]
                    epochs=dt["epochs"]
                    ma=dt["ma"]
                    ma1=dt["ma1"]
                    if ty==3:
                        max_td=dt["max_td"]
                        predict_future=dt["predict_future"]
                        predict_next=dt["predict_next"]
                        timeseries_data=np.array(dt["timeseries_data"])

                    op=dt["op"]
                    pre=dt["pre"]
                    if pre!=[] and pre!="":
                        pre=np.array(dt["pre"])
                        try:
                            pred=dt["pred"]
                        except:
                            pred=""
                    preee=dt["preee"]
                    pree=dt["pree"]

                    uni_y=len(np.unique(target))
                    lstm=[]
                    if n_hidden_layers>1:
                        for i in range(n_hidden_layers):
                            if ty==0:
                                if i!=0:
                                    if i!=n_hidden_layers-1:
                                        lstm.append(SimpleRNN(1,activation=activation[act],use_bias=use_bias,return_sequences=True, dropout=dropout))
                                    else:
                                        lstm.append(SimpleRNN(1,activation=activation[act],use_bias=use_bias,return_sequences=False, dropout=dropout))
                                else:
                                    lstm.append(SimpleRNN(1, batch_input_shape=(None,None,data.shape[2]),activation=activation[act],use_bias=use_bias,return_sequences=True, dropout=dropout))
                            elif ty==1:
                                if i!=0:
                                    if i!=n_hidden_layers-1:
                                        lstm.append(SimpleRNN(no_of_units[i],activation=activation[act],use_bias=use_bias,return_sequences=True, dropout=dropout))
                                    else:
                                        lstm.append(SimpleRNN(no_of_units[i],activation=activation[act],use_bias=use_bias,return_sequences=False, dropout=dropout))
                                else:
                                    lstm.append(SimpleRNN(no_of_units[i], batch_input_shape=(None,None,data.shape[2]),activation=activation[act],use_bias=use_bias,return_sequences=True, dropout=dropout))
                            elif ty==2:
                                if i!=0:
                                    if i!=n_hidden_layers-1:
                                        lstm.append(SimpleRNN(no_of_units[i],activation=activation[act],use_bias=use_bias,return_sequences=True, dropout=dropout))
                                    else:
                                        lstm.append(SimpleRNN(no_of_units[i],activation=activation[act],use_bias=use_bias,return_sequences=False, dropout=dropout))
                                else:
                                    lstm.append(SimpleRNN(no_of_units[i],activation=activation[act],use_bias=use_bias,return_sequences=True, dropout=dropout))
                            elif ty==3:
                                if i!=0:
                                    if i!=n_hidden_layers-1:
                                        lstm.append(SimpleRNN(no_of_units[i],activation=activation[act],return_sequences=True,use_bias=use_bias, dropout=dropout))
                                    else:
                                        lstm.append(SimpleRNN(no_of_units[i],activation=activation[act],return_sequences=False,use_bias=use_bias, dropout=dropout))
                                else:
                                    lstm.append(SimpleRNN(no_of_units[i], input_shape=(time_steps, n_features),activation=activation[act],return_sequences=True,use_bias=use_bias, dropout=dropout))
                    else:
                        if ty==0:
                            lstm.append(SimpleRNN(1, batch_input_shape=(None,None,data.shape[2]),use_bias=use_bias,return_sequences=False, dropout=dropout))
                        elif ty==1:
                            lstm.append(SimpleRNN(no_of_units[0],batch_input_shape=(None,None,data.shape[2]),activation=activation[act],use_bias=use_bias,return_sequences=False, dropout=dropout))
                        elif ty==2:
                            lstm.append(SimpleRNN(no_of_units[0],activation=activation[act],use_bias=use_bias,return_sequences=False, dropout=dropout))
                        elif ty==3:
                            lstm.append(SimpleRNN(no_of_units[0],activation=activation[act],input_shape=(time_steps, n_features),use_bias=use_bias,return_sequences=False, dropout=dropout))

                    if ty!=2:
                        model = Sequential(lstm)
                    else:
                        model = Sequential()
                        model.add(Embedding(2000,no_of_features,input_length=data.shape[1]))
                        for i in lstm:
                            model.add(i)
                    if add_dense_layer==0:
                        if ty==0 or ty==3:
                            for i in range(no_of_hidden_dense_layer):
                                model.add(Dense(no_of_dense_layer_neurons[i],activation=D_act[dc]))
                            model.add(Dense(f,activation=o_act[oc]))
                        elif ty==1:
                            for i in range(no_of_hidden_dense_layer):
                                model.add(Dense(no_of_dense_layer_neurons[i],activation=D_act[dc]))
                            model.add(Dense(uni_y,activation=o_act[oc]))
                        elif ty==2:
                            for i in range(no_of_hidden_dense_layer):
                                model.add(Dense(no_of_dense_layer_neurons[i],activation=D_act[dc]))
                            model.add(Dense(2,activation=o_act[oc]))
                    model.compile(loss=loss[los], optimizer=optimizer[op],metrics=[metrics[mt]])   
                    if test_train==0:
                        his=model.fit(data, target, epochs=epochs, shuffle=False,verbose=0)
                    else:
                        his=model.fit(x_train, y_train, epochs=epochs, shuffle=False,verbose=0)


                    if test_train==0:
                        score=model.evaluate(data,target,verbose=0)
                        output+='Trained data Lose : '+str(score[0])+"\n\n"
                    else:
                        score=model.evaluate(x_test,y_test,verbose=0)
                        output+='Tested data Lose : '+str(score[0])+"\n\n"
                    if ty==1 or ty==2:
                            if test_train==0:
                                    output+='Trained data Accuracy : '+str(score[1])+"\n\n"
                            else:
                                    output+='Tested data Accuracy : '+str(score[1])+"\n\n"

                    if test_train==0 and ty!=3:
                        fig=plt.figure(facecolor='lightgreen')
                        ax=plt.axes()
                        ax.set_facecolor('lightgreen')
                        ay=fig.add_subplot(1,1,1)
                        plt.title("Target(green) vs predicted(Blue)")
                        if ty==0:
                            predict = model.predict(data)
                            ay.scatter(range(len(data)),predict,c='r')
                        else:
                            pret=model.predict(data)
                            predict = np.argmax(pret,axis=1)
                            b=[pret[i][predict[i]] for i in range(len(predict))]
                            ay.scatter(range(len(data)),b,c='r')
                        ay.scatter(range(len(data)),target,c='g')
                        fig.canvas.draw()
                        img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                        img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                        pil_im=Image.fromarray(img)
                        buff=io.BytesIO()
                        pil_im.save(buff,format="PNG")
                        img_str=base64.b64encode(buff.getvalue())
                        
                        if ty==0:
                            p=np.ceil(((model.predict(data).reshape(1,len(data))[0])*ma))
                            output+='Accuracy : '+str(sum([(target[i]*ma1)==p[i] for i in range(len(data))])/len(data))+"\n\n"
                        else:
                                pred=model.predict(data)
                                predict = np.argmax(pred,axis=1)
                        if(len(data)<=20):
                            if ty==0:
                                output+="Predicted "+str(sum(((target*ma1)==p)))+" Correctly Out of "+str(len(target))+"\n\n"
                                output+="predicted output value of trained data:\t"+str(p)+"\n\n"
                            elif ty==1:
                                output+="Predicted "+str(sum((target==predict)))+" Correctly Out of "+str(len(target))+"\n\n"
                                output+="predicted output value of trained data:\t"+str(predict)+"\n\n"
                            elif ty==2:
                                output+="Predicted "+str(sum((target==predict)))+" Correctly Out of "+str(len(target[:-1]))+"\n\n"
                                output+="predicted output value of trained data:\t"+str(predict[:-1])+"\n\n"
                        else:
                            if ty==0:
                                output+="Predicted "+str(sum(((target*ma1)==p)))+" Correctly Out of "+str(len(target))+"\n\n"
                                output+="predicted first 20 sample output value of trained data:\t"+str(np.ceil((model.predict(data[:20]).reshape(1,20)[0])*ma))+"\n\n"
                            elif ty==1:
                                output+="Predicted "+str(sum((target==predict)))+" Correctly Out of "+str(len(target))+"\n\n"
                                output+="predicted first 20 sample output value of trained data:\t"+str(predict[:20])+"\n\n"
                            elif ty==2:
                                output+="Predicted "+str(sum((target==predict)))+" Correctly Out of "+str(len(target[:-1]))+"\n\n"
                                output+="predicted output value of trained data:\t"+str(predict[:-1])+"\n\n"
                    

                    elif ty!=3:
                        fig=plt.figure(facecolor='lightgreen')
                        ax=plt.axes()
                        ax.set_facecolor('lightgreen')
                        ay=fig.add_subplot(1,1,1)
                        if ty==0:
                            predict = model.predict(x_test)
                            ay.scatter(range(len(x_test)),predict,c='r')
                        else:
                            pret=model.predict(x_test)
                            predict = np.argmax(pret,axis=1)
                            b=[pret[i][predict[i]] for i in range(len(predict))]
                            ay.scatter(range(len(x_test)),b,c='r')
                        plt.title("y_test(green) vs y_predicted(Blue)")
                        ay.scatter(range(len(x_test)),y_test,c='g')
                        fig.canvas.draw()
                        img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                        img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                        pil_im=Image.fromarray(img)
                        buff=io.BytesIO()
                        pil_im.save(buff,format="PNG")
                        img_str=base64.b64encode(buff.getvalue())
                        if ty==0:
                            p=np.ceil(((model.predict(x_test).reshape(1,len(x_test))[0])*ma))
                            pa=np.ceil(((model.predict(x_train).reshape(1,len(x_train))[0])*ma))
                            output+='Accuracy : '+str(sum([(y_test[i]*ma1)==p[i] for i in range(len(x_test))])/len(x_test))+"\n\n"
                        else:
                            predict = np.argmax(model.predict(x_test),axis=1)
                            predic = np.argmax(model.predict(x_train),axis=1)
                        if(len(x_train)<=20):
                            if ty==0:
                                output+="Predicted "+str(sum(((y_test*ma1)==p)))+" Correctly Out of "+str(len(y_test))+"\n\n"
                                output+="predicted output value of trained data:\t"+str(pa)+"\n\n"
                            else:
                                output+="Predicted "+str(sum((y_test==predict)))+" Correctly Out of "+str(len(y_test))+"\n\n"
                                output+="predicted output value of trained data:\t"+str(predic)+"\n\n"
                        else:
                            if ty==0:
                                output+="Predicted "+str(sum(((y_test*ma1)==p)))+" Correctly Out of "+str(len(y_test))+"\n\n"
                                output+="predicted first 20 sample output value of trained data:\t"+str(np.ceil((model.predict(x_train[:20]).reshape(1,20)[0])*ma))+"\n\n"
                            else:
                                output+="Predicted "+str(sum((y_test==predict)))+" Correctly Out of "+str(len(y_test))+"\n\n"
                                output+="predicted first 20 sample output value of trained data:\t"+str(predic[:20])+"\n\n"
                    fig=plt.figure(facecolor='lightgreen')
                    ax=plt.axes()
                    ax.set_facecolor('lightgreen')
                    ay=fig.add_subplot(1,1,1)            
                    plt.title("Loss")
                    ay.plot(his.history['loss'])
                    fig.canvas.draw()
                    img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                    img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                    pil_im=Image.fromarray(img)
                    buff=io.BytesIO()
                    pil_im.save(buff,format="PNG")
                    img_str2=base64.b64encode(buff.getvalue())
                    
                    if pree or preee:
                        if ty==0:
                            pre1=model.predict(pre).reshape(1,len(pred))[0]
                            output+="Predicted output of user's input :"+str(np.ceil((pre1*ma)))+"\n\n"
                        elif ty==1:
                            pre1=model.predict(pre)
                            output+="Predicted output of user's input :"+str(np.argmax(pre1,axis=1))+"\n\n"
                        elif ty==2:
                            pre1=model.predict(pre)
                            output+="Predicted output of user's input :"+str(np.argmax(pre1,axis=1))+"\n\n"
                        elif ty==3:
                                try:
                                        a=model.predict(pre)
                                        if n_features==1 or f==1:
                                                trained_pre=[i[0] for i in a]
                                        else:
                                                trained_pre=a.tolist()
                                        output+="Predicted output of user's input :"+str(trained_pre)+"\n\n"
                                except:
                                        output+="Please check your input for Prediction. The time_steps number of data requirerd to predict one output and requiers all trained features to predict future."+"\n\n"

                    if ty==3:
                        target=target*ma1
                        if test_train==0:
                                a=model.predict(data)*ma1
                        else:
                                a=model.predict(x_train)*ma1
                        if n_features==1 or f==1:
                                trained_pre=[i[0] for i in a]
                        else:
                                trained_pre=a.tolist()
                        if len(data)<=20:
                                output+="Predicted output of trained data :\t"+str(trained_pre)+"\n\n"
                        else:
                                output+="Predicted first 20 sample output value of trained data :\t"+str(trained_pre[:20])+"\n\n"
                        if n_features==1 or f==1:
                                yy=[i[0][0] for i in target]
                        else:
                                yy=np.array([target[i][0] for i in range(len(target))])
                        fig=plt.figure(facecolor='lightgreen')
                        ax=plt.axes()
                        ax.set_facecolor('lightgreen')
                        if f>1:
                                for i in range(1,len(a[0])+1):
                                        plt.subplot(len(a[0])-1,2,i)
                                        plt.title("Column-"+str(i)+" series")
                                        plt.plot(yy[:,i-1:i],c='green')
                                        plt.plot(a[:,i-1:i],c='red')
                        else:
                                plt.plot(yy,c='green')
                                plt.plot(a,c='red')
                        plt.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.9)
                        plt.suptitle("Trained(green) vs Predicted(blue)")
                        fig.canvas.draw()
                        img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                        img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                        pil_im=Image.fromarray(img)
                        buff=io.BytesIO()
                        pil_im.save(buff,format="PNG")
                        img_str22=base64.b64encode(buff.getvalue())


                        if test_train==1:
                                y_test=y_test*ma1
                                a=model.predict(x_test)*ma1
                                fig=plt.figure(facecolor='lightgreen')
                                ax=plt.axes()
                                ax.set_facecolor('lightgreen')
                                if n_features==1 or f==1:
                                        trained_pre=[i[0] for i in a]
                                else:
                                        trained_pre=a.tolist()
                                if len(data)<=20:
                                        output+="Predicted output of tested data :"+str(trained_pre)+"\n\n"
                                else:
                                        output+="Predicted first 20 sample output value of tested data :\t"+str(trained_pre[:20])+"\n\n"
                                if n_features==1 or f==1:
                                        yy=[i[0][0] for i in y_test]
                                else:
                                        yy=np.array([y_test[i][0] for i in range(len(y_test))])
                                if f>1:
                                        for i in range(1,len(a[0])+1):
                                                plt.subplot(len(a[0])-1,2,i)
                                                plt.title("Column-"+str(i)+" series")
                                                plt.plot(yy[:,i-1:i],c='green')
                                                plt.plot(a[:,i-1:i],c='red')
                                else:
                                        plt.plot(yy,c='green')
                                        plt.plot(a,c='red')
                                plt.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.9)
                                plt.suptitle("y_test(green) vs y_Predicted(blue)")
                                fig.canvas.draw()
                                img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                                img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                                img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                                pil_im=Image.fromarray(img)
                                buff=io.BytesIO()
                                pil_im.save(buff,format="PNG")
                                img_str24=base64.b64encode(buff.getvalue())
                                
                        if predict_future==1:
                                x_input = np.array(timeseries_data[len(timeseries_data)-time_steps:])
                                temp_input=list(x_input)
                                lst_output=[]
                                i=0
                                while(i<predict_next):
                                    if(len(temp_input)>time_steps):
                                        x_input=np.array(temp_input[1:])
                                        x_input = x_input.reshape((1, time_steps, n_features))
                                        yhat = model.predict(x_input, verbose=0)
                                        if n_features>1:
                                                temp_input.append(yhat[0])
                                        else:
                                                temp_input.append(yhat[0][0])
                                        temp_input=temp_input[1:]
                                        if n_features>1:
                                                lst_output.append(yhat[0])
                                        else:
                                                lst_output.append(yhat[0][0])
                                        i=i+1
                                    else:
                                        x_input = x_input.reshape((1, time_steps, n_features))
                                        yhat = model.predict(x_input, verbose=0)
                                        if n_features>1:
                                                temp_input.append(yhat[0])
                                                lst_output.append(yhat[0])
                                        else:
                                                temp_input.append(yhat[0][0])
                                                lst_output.append(yhat[0][0])
                                        i=i+1

                                output+="Predicted next "+str(predict_next)+" values :"+"\n\n"

                                lst_output=np.array(lst_output)*ma1
                                output+=str(lst_output)+"\n\n"

                                timeseries_data*=max_td
                                day_new=np.arange(1,len(timeseries_data)+1)
                                day_pred=np.arange(len(timeseries_data)+1,len(timeseries_data)+predict_next+1)
                                if f>1:
                                        fig=plt.figure(facecolor='lightgreen')
                                        ax=plt.axes()
                                        ax.set_facecolor('lightgreen')
                                        for i in range(1,len(timeseries_data[0])+1):
                                                plt.subplot(len(timeseries_data[0])-1,2,i)
                                                plt.title("Column-"+str(i)+" series")
                                                plt.plot(day_new,np.array(timeseries_data)[:,i-1:i],c='green')
                                                plt.plot(day_pred,np.array(lst_output)[:,i-1:i],c='red')
                                        plt.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.9)
                                else:
                                        plt.plot(day_new,timeseries_data,c='green')
                                        plt.plot(day_pred,lst_output,c='red')
                                plt.suptitle("Trained(green) vs Predicted future "+str(predict_next)+" values(blue)")
                                fig.canvas.draw()
                                img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                                img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                                img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                                pil_im=Image.fromarray(img)
                                buff=io.BytesIO()
                                pil_im.save(buff,format="PNG")
                                img_str23=base64.b64encode(buff.getvalue())
                                
                    output+="\n"
                    if ty!=3:
                            #return output,""+str(img_str2,'utf-8'),""+str(img_str,'utf-8'),"",""
                            return JsonResponse([output,""+str(img_str2,'utf-8'),""+str(img_str,'utf-8'),"",""],safe=False)
                    else:
                            if predict_future==1 and test_train==1:
                                    #return output,""+str(img_str2,'utf-8'),""+str(img_str22,'utf-8'),""+str(img_str24,'utf-8'),""+str(img_str23,'utf-8'),""
                                    return JsonResponse([output,""+str(img_str2,'utf-8'),""+str(img_str22,'utf-8'),""+str(img_str24,'utf-8'),""+str(img_str23,'utf-8'),""],safe=False)
                            elif predict_future==0 and test_train==1:
                                #return output,""+str(img_str2,'utf-8'),""+str(img_str22,'utf-8'),""+str(img_str24,'utf-8'),"",""
                                return JsonResponse([output,""+str(img_str2,'utf-8'),""+str(img_str22,'utf-8'),""+str(img_str24,'utf-8'),"",""],safe=False)
                            elif predict_future==1 and test_train==0:
                                #return output,""+str(img_str2,'utf-8'),""+str(img_str22,'utf-8'),"",""+str(img_str23,'utf-8'),""
                                return JsonResponse([output,""+str(img_str2,'utf-8'),""+str(img_str22,'utf-8'),"",""+str(img_str23,'utf-8'),""],safe=False)
                            else:
                                    #return output,""+str(img_str2,'utf-8'),""+str(img_str22,'utf-8'),"","","" 
                                    return JsonResponse([output,""+str(img_str2,'utf-8'),""+str(img_str22,'utf-8'),"","",""],safe=False)
                                
    except Exception as e:
        #return Response(e.args[0],status.HTTP_400_BAD_REQUEST)                    
        exception_type, exception_object, exception_traceback = sys.exc_info()
        line_number = exception_traceback.tb_lineno
        output="Something went wrong... Please give correct input"
        return JsonResponse(str(line_number)+str(exception_type)+" "+str(e)+"error",safe=False)



@api_view(["POST"])
def api4(x):
    try:
                    output=""
                    dt=json.loads(x.body)
                    test_train=dt["test_train"]
                    re_act=dt["re_act"]
                    data=np.array(dt["data"])
                    target=np.array(dt["target"])
                    if test_train==1:
                        x_test=np.array(dt["x_test"])
                        y_test=np.array(dt["y_test"])
                        x_train=np.array(dt["x_train"])
                        y_train=np.array(dt["y_train"])
                        target=np.array(dt["target"])
                    loss=['mse','mean_absolute_error','categorical_crossentropy','binary_crossentropy']
                    optimizer=['adam','sgd','RMSprop','Adadelta','Adagrad','Adamax','Nadam','Ftrl']
                    metrics=['accuracy','AUC','Precision','Recall']
                    activation=['tanh','sigmoid','relu','softmax']
                    rec_activation=['sigmoid','tanh','relu','softmax']
                    o_act=['relu','softmax','sigmoid']
                    D_act=['relu','sigmoid','softmax']

                    act=dt["act"]
                    use_bias=dt["use_bias"]
                    if use_bias==0:
                        use_bias=True
                    else:
                        use_bias=False
                    n_hidden_layers=dt["n_hidden_layers"]
                    ty=dt["ty"]
                    dropout=dt["dropout"]
                    no_of_units=dt["no_of_units"]
                    if ty==3:
                        time_steps=dt["time_steps"]
                        n_features=dt["n_features"]
                    if ty==2:
                        no_of_features=dt["no_of_features"]
                    add_dense_layer=dt["add_dense_layer"]
                    no_of_hidden_dense_layer=dt["no_of_hidden_dense_layer"]
                    no_of_dense_layer_neurons=dt["no_of_dense_layer_neurons"]
                    oc=dt["oc"]
                    dc=dt["dc"]
                    if ty==0 or ty==3:
                        f=dt["f"]
                    los=dt["los"]
                    mt=dt["mt"]
                    epochs=dt["epochs"]
                    ma=dt["ma"]
                    ma1=dt["ma1"]
                    if ty==3:
                        max_td=dt["max_td"]
                        predict_future=dt["predict_future"]
                        predict_next=dt["predict_next"]
                        timeseries_data=np.array(dt["timeseries_data"])

                    op=dt["op"]
                    pre=dt["pre"]
                    if pre!=[] and pre!="":
                        pre=np.array(dt["pre"])
                        try:
                            pred=dt["pred"]
                        except:
                            pred=""
                    preee=dt["preee"]
                    pree=dt["pree"]

                    uni_y=len(np.unique(target))
                    lstm=[]
                    if n_hidden_layers>1:
                        for i in range(n_hidden_layers):
                            if ty==0:
                                if i!=0:
                                    if i!=n_hidden_layers-1:
                                        lstm.append(LSTM(1,activation=activation[act],dropout=dropout,recurrent_activation=rec_activation[re_act],use_bias=use_bias,return_sequences=True))
                                    else:
                                        lstm.append(LSTM(1,activation=activation[act],dropout=dropout,recurrent_activation=rec_activation[re_act],use_bias=use_bias,return_sequences=False))
                                else:
                                    lstm.append(LSTM(1, batch_input_shape=(None,None,data.shape[2]),dropout=dropout,activation=activation[act],recurrent_activation=rec_activation[re_act],use_bias=use_bias,return_sequences=True))
                            elif ty==1:
                                if i!=0:
                                    if i!=n_hidden_layers-1:
                                        lstm.append(LSTM(no_of_units[i],activation=activation[act],dropout=dropout,recurrent_activation=rec_activation[re_act],use_bias=use_bias,return_sequences=True))
                                    else:
                                        lstm.append(LSTM(no_of_units[i],activation=activation[act],dropout=dropout,recurrent_activation=rec_activation[re_act],use_bias=use_bias,return_sequences=False))
                                else:
                                    lstm.append(LSTM(no_of_units[i], batch_input_shape=(None,None,data.shape[2]),dropout=dropout,activation=activation[act],recurrent_activation=rec_activation[re_act],use_bias=use_bias,return_sequences=True))
                            elif ty==2:
                                if i!=0:
                                    if i!=n_hidden_layers-1:
                                        lstm.append(LSTM(no_of_units[i],activation=activation[act],dropout=dropout,recurrent_activation=rec_activation[re_act],use_bias=use_bias,return_sequences=True))
                                    else:
                                        lstm.append(LSTM(no_of_units[i],activation=activation[act],dropout=dropout,recurrent_activation=rec_activation[re_act],use_bias=use_bias,return_sequences=False))
                                else:
                                    lstm.append(LSTM(no_of_units[i],recurrent_activation=rec_activation[re_act],dropout=dropout,activation=activation[act],use_bias=use_bias,return_sequences=True))
                            elif ty==3:
                                if i!=0:
                                    if i!=n_hidden_layers-1:
                                        lstm.append(LSTM(no_of_units[i],activation=activation[act],dropout=dropout,return_sequences=True,recurrent_activation=rec_activation[re_act],use_bias=use_bias))
                                    else:
                                        lstm.append(LSTM(no_of_units[i],activation=activation[act],dropout=dropout,return_sequences=False,recurrent_activation=rec_activation[re_act],use_bias=use_bias))
                                else:
                                    lstm.append(LSTM(no_of_units[i], input_shape=(time_steps, n_features),dropout=dropout,activation=activation[act],return_sequences=True,recurrent_activation=rec_activation[re_act],use_bias=use_bias))
                    else:
                        if ty==0:
                            lstm.append(LSTM(1,batch_input_shape=(None,None,data.shape[2]),dropout=dropout,recurrent_activation=rec_activation[re_act],use_bias=use_bias,return_sequences=False))
                        elif ty==1:
                            lstm.append(LSTM(no_of_units[0],batch_input_shape=(None,None,data.shape[2]),dropout=dropout,activation=activation[act],recurrent_activation=rec_activation[re_act],use_bias=use_bias,return_sequences=False))
                        elif ty==2:
                            lstm.append(LSTM(no_of_units[0],activation=activation[act],dropout=dropout,recurrent_activation=rec_activation[re_act],use_bias=use_bias,return_sequences=False))
                        elif ty==3:
                            lstm.append(LSTM(no_of_units[0],activation=activation[act],dropout=dropout,input_shape=(time_steps, n_features),recurrent_activation=rec_activation[re_act],use_bias=use_bias,return_sequences=False))

                    if ty!=2:
                        model = Sequential(lstm)
                    else:
                        model = Sequential()
                        model.add(Embedding(2000,no_of_features,input_length=data.shape[1]))
                        for i in lstm:
                            model.add(i)
                    if add_dense_layer==0:
                        if ty==0 or ty==3:
                            for i in range(no_of_hidden_dense_layer):
                                model.add(Dense(no_of_dense_layer_neurons[i],activation=D_act[dc]))
                            model.add(Dense(f,activation=o_act[oc]))
                        elif ty==1:
                            for i in range(no_of_hidden_dense_layer):
                                model.add(Dense(no_of_dense_layer_neurons[i],activation=D_act[dc]))
                            model.add(Dense(uni_y,activation=o_act[oc]))
                        elif ty==2:
                            for i in range(no_of_hidden_dense_layer):
                                model.add(Dense(no_of_dense_layer_neurons[i],activation=D_act[dc]))
                            model.add(Dense(2,activation=o_act[oc]))
                    model.compile(loss=loss[los], optimizer=optimizer[op],metrics=[metrics[mt]])   
                    if test_train==0:
                        his=model.fit(data, target, epochs=epochs, shuffle=False,verbose=0)
                    else:
                        his=model.fit(x_train, y_train, epochs=epochs, shuffle=False,verbose=0)


                    if test_train==0:
                        score=model.evaluate(data,target,verbose=0)
                        output+='Trained data Lose : '+str(score[0])+"\n\n"
                    else:
                        score=model.evaluate(x_test,y_test,verbose=0)
                        output+='Tested data Lose : '+str(score[0])+"\n\n"
                    if ty==1 or ty==2:
                            if test_train==0:
                                    output+='Trained data Accuracy : '+str(score[1])+"\n\n"
                            else:
                                    output+='Tested data Accuracy : '+str(score[1])+"\n\n"

                    if test_train==0 and ty!=3:
                        fig=plt.figure(facecolor='lightgreen')
                        ax=plt.axes()
                        ax.set_facecolor('lightgreen')
                        ay=fig.add_subplot(1,1,1)
                        plt.title("Target(green) vs predicted(Blue)")
                        if ty==0:
                            predict = model.predict(data)
                            ay.scatter(range(len(data)),predict,c='r')
                        else:
                            pret=model.predict(data)
                            predict = np.argmax(pret,axis=1)
                            b=[pret[i][predict[i]] for i in range(len(predict))]
                            ay.scatter(range(len(data)),b,c='r')
                        ay.scatter(range(len(data)),target,c='g')
                        fig.canvas.draw()
                        img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                        img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                        pil_im=Image.fromarray(img)
                        buff=io.BytesIO()
                        pil_im.save(buff,format="PNG")
                        img_str=base64.b64encode(buff.getvalue())
                        
                        if ty==0:
                            p=np.ceil(((model.predict(data).reshape(1,len(data))[0])*ma))
                            output+='Accuracy : '+str(sum([(target[i]*ma1)==p[i] for i in range(len(data))])/len(data))+"\n\n"
                        else:
                                pred=model.predict(data)
                                predict = np.argmax(pred,axis=1)
                        if(len(data)<=20):
                            if ty==0:
                                output+="Predicted "+str(sum(((target*ma1)==p)))+" Correctly Out of "+str(len(target))+"\n\n"
                                output+="predicted output value of trained data:\t"+str(p)+"\n\n"
                            elif ty==1:
                                output+="Predicted "+str(sum((target==predict)))+" Correctly Out of "+str(len(target))+"\n\n"
                                output+="predicted output value of trained data:\t"+str(predict)+"\n\n"
                            elif ty==2:
                                output+="Predicted "+str(sum((target==predict)))+" Correctly Out of "+str(len(target[:-1]))+"\n\n"
                                output+="predicted output value of trained data:\t"+str(predict[:-1])+"\n\n"
                        else:
                            if ty==0:
                                output+="Predicted "+str(sum(((target*ma1)==p)))+" Correctly Out of "+str(len(target))+"\n\n"
                                output+="predicted first 20 sample output value of trained data:\t"+str(np.ceil((model.predict(data[:20]).reshape(1,20)[0])*ma))+"\n\n"
                            elif ty==1:
                                output+="Predicted "+str(sum((target==predict)))+" Correctly Out of "+str(len(target))+"\n\n"
                                output+="predicted first 20 sample output value of trained data:\t"+str(predict[:20])+"\n\n"
                            elif ty==2:
                                output+="Predicted "+str(sum((target==predict)))+" Correctly Out of "+str(len(target[:-1]))+"\n\n"
                                output+="predicted output value of trained data:\t"+str(predict[:-1])+"\n\n"
                    

                    elif ty!=3:
                        fig=plt.figure(facecolor='lightgreen')
                        ax=plt.axes()
                        ax.set_facecolor('lightgreen')
                        ay=fig.add_subplot(1,1,1)
                        if ty==0:
                            predict = model.predict(x_test)
                            ay.scatter(range(len(x_test)),predict,c='r')
                        else:
                            pret=model.predict(x_test)
                            predict = np.argmax(pret,axis=1)
                            b=[pret[i][predict[i]] for i in range(len(predict))]
                            ay.scatter(range(len(x_test)),b,c='r')
                        plt.title("y_test(green) vs y_predicted(Blue)")
                        ay.scatter(range(len(x_test)),y_test,c='g')
                        fig.canvas.draw()
                        img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                        img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                        pil_im=Image.fromarray(img)
                        buff=io.BytesIO()
                        pil_im.save(buff,format="PNG")
                        img_str=base64.b64encode(buff.getvalue())
                        if ty==0:
                            p=np.ceil(((model.predict(x_test).reshape(1,len(x_test))[0])*ma))
                            pa=np.ceil(((model.predict(x_train).reshape(1,len(x_train))[0])*ma))
                            output+='Accuracy : '+str(sum([(y_test[i]*ma1)==p[i] for i in range(len(x_test))])/len(x_test))+"\n\n"
                        else:
                            predict = np.argmax(model.predict(x_test),axis=1)
                            predic = np.argmax(model.predict(x_train),axis=1)
                        if(len(x_train)<=20):
                            if ty==0:
                                output+="Predicted "+str(sum(((y_test*ma1)==p)))+" Correctly Out of "+str(len(y_test))+"\n\n"
                                output+="predicted output value of trained data:\t"+str(pa)+"\n\n"
                            else:
                                output+="Predicted "+str(sum((y_test==predict)))+" Correctly Out of "+str(len(y_test))+"\n\n"
                                output+="predicted output value of trained data:\t"+str(predic)+"\n\n"
                        else:
                            if ty==0:
                                output+="Predicted "+str(sum(((y_test*ma1)==p)))+" Correctly Out of "+str(len(y_test))+"\n\n"
                                output+="predicted first 20 sample output value of trained data:\t"+str(np.ceil((model.predict(x_train[:20]).reshape(1,20)[0])*ma))+"\n\n"
                            else:
                                output+="Predicted "+str(sum((y_test==predict)))+" Correctly Out of "+str(len(y_test))+"\n\n"
                                output+="predicted first 20 sample output value of trained data:\t"+str(predic[:20])+"\n\n"
                    fig=plt.figure(facecolor='lightgreen')
                    ax=plt.axes()
                    ax.set_facecolor('lightgreen')
                    ay=fig.add_subplot(1,1,1)            
                    plt.title("Loss")
                    ay.plot(his.history['loss'])
                    fig.canvas.draw()
                    img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                    img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                    pil_im=Image.fromarray(img)
                    buff=io.BytesIO()
                    pil_im.save(buff,format="PNG")
                    img_str2=base64.b64encode(buff.getvalue())
                    
                    if pree or preee:
                        if ty==0:
                            pre1=model.predict(pre).reshape(1,len(pred))[0]
                            output+="Predicted output of user's input :"+str(np.ceil((pre1*ma)))+"\n\n"
                        elif ty==1:
                            pre1=model.predict(pre)
                            output+="Predicted output of user's input :"+str(np.argmax(pre1,axis=1))+"\n\n"
                        elif ty==2:
                            pre1=model.predict(pre)
                            output+="Predicted output of user's input :"+str(np.argmax(pre1,axis=1))+"\n\n"
                        elif ty==3:
                                try:
                                        a=model.predict(pre)
                                        if n_features==1 or f==1:
                                                trained_pre=[i[0] for i in a]
                                        else:
                                                trained_pre=a.tolist()
                                        output+="Predicted output of user's input :"+str(trained_pre)+"\n\n"
                                except:
                                        output+="Please check your input for Prediction. The time_steps number of data requirerd to predict one output and requiers all trained features to predict future."+"\n\n"

                    if ty==3:
                        target=target*ma1
                        if test_train==0:
                                a=model.predict(data)*ma1
                        else:
                                a=model.predict(x_train)*ma1
                        if n_features==1 or f==1:
                                trained_pre=[i[0] for i in a]
                        else:
                                trained_pre=a.tolist()
                        if len(data)<=20:
                                output+="Predicted output of trained data :\t"+str(trained_pre)+"\n\n"
                        else:
                                output+="Predicted first 20 sample output value of trained data :\t"+str(trained_pre[:20])+"\n\n"
                        if n_features==1 or f==1:
                                yy=[i[0][0] for i in target]
                        else:
                                yy=np.array([target[i][0] for i in range(len(target))])
                        fig=plt.figure(facecolor='lightgreen')
                        ax=plt.axes()
                        ax.set_facecolor('lightgreen')
                        if f>1:
                                for i in range(1,len(a[0])+1):
                                        plt.subplot(len(a[0])-1,2,i)
                                        plt.title("Column-"+str(i)+" series")
                                        plt.plot(yy[:,i-1:i],c='green')
                                        plt.plot(a[:,i-1:i],c='red')
                        else:
                                plt.plot(yy,c='green')
                                plt.plot(a,c='red')
                        plt.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.9)
                        plt.suptitle("Trained(green) vs Predicted(blue)")
                        fig.canvas.draw()
                        img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                        img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                        pil_im=Image.fromarray(img)
                        buff=io.BytesIO()
                        pil_im.save(buff,format="PNG")
                        img_str22=base64.b64encode(buff.getvalue())


                        if test_train==1:
                                y_test=y_test*ma1
                                a=model.predict(x_test)*ma1
                                fig=plt.figure(facecolor='lightgreen')
                                ax=plt.axes()
                                ax.set_facecolor('lightgreen')
                                if n_features==1 or f==1:
                                        trained_pre=[i[0] for i in a]
                                else:
                                        trained_pre=a.tolist()
                                if len(data)<=20:
                                        output+="Predicted output of tested data :"+str(trained_pre)+"\n\n"
                                else:
                                        output+="Predicted first 20 sample output value of tested data :\t"+str(trained_pre[:20])+"\n\n"
                                if n_features==1 or f==1:
                                        yy=[i[0][0] for i in y_test]
                                else:
                                        yy=np.array([y_test[i][0] for i in range(len(y_test))])
                                if f>1:
                                        for i in range(1,len(a[0])+1):
                                                plt.subplot(len(a[0])-1,2,i)
                                                plt.title("Column-"+str(i)+" series")
                                                plt.plot(yy[:,i-1:i],c='green')
                                                plt.plot(a[:,i-1:i],c='red')
                                else:
                                        plt.plot(yy,c='green')
                                        plt.plot(a,c='red')
                                plt.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.9)
                                plt.suptitle("y_test(green) vs y_Predicted(blue)")
                                fig.canvas.draw()
                                img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                                img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                                img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                                pil_im=Image.fromarray(img)
                                buff=io.BytesIO()
                                pil_im.save(buff,format="PNG")
                                img_str24=base64.b64encode(buff.getvalue())
                                
                        if predict_future==1:
                                x_input = np.array(timeseries_data[len(timeseries_data)-time_steps:])
                                temp_input=list(x_input)
                                lst_output=[]
                                i=0
                                while(i<predict_next):
                                    if(len(temp_input)>time_steps):
                                        x_input=np.array(temp_input[1:])
                                        x_input = x_input.reshape((1, time_steps, n_features))
                                        yhat = model.predict(x_input, verbose=0)
                                        if n_features>1:
                                                temp_input.append(yhat[0])
                                        else:
                                                temp_input.append(yhat[0][0])
                                        temp_input=temp_input[1:]
                                        if n_features>1:
                                                lst_output.append(yhat[0])
                                        else:
                                                lst_output.append(yhat[0][0])
                                        i=i+1
                                    else:
                                        x_input = x_input.reshape((1, time_steps, n_features))
                                        yhat = model.predict(x_input, verbose=0)
                                        if n_features>1:
                                                temp_input.append(yhat[0])
                                                lst_output.append(yhat[0])
                                        else:
                                                temp_input.append(yhat[0][0])
                                                lst_output.append(yhat[0][0])
                                        i=i+1

                                output+="Predicted next "+str(predict_next)+" values :"+"\n\n"

                                lst_output=np.array(lst_output)*ma1
                                output+=str(lst_output)+"\n\n"

                                timeseries_data*=max_td
                                day_new=np.arange(1,len(timeseries_data)+1)
                                day_pred=np.arange(len(timeseries_data)+1,len(timeseries_data)+predict_next+1)
                                if f>1:
                                        fig=plt.figure(facecolor='lightgreen')
                                        ax=plt.axes()
                                        ax.set_facecolor('lightgreen')
                                        for i in range(1,len(timeseries_data[0])+1):
                                                plt.subplot(len(timeseries_data[0])-1,2,i)
                                                plt.title("Column-"+str(i)+" series")
                                                plt.plot(day_new,np.array(timeseries_data)[:,i-1:i],c='green')
                                                plt.plot(day_pred,np.array(lst_output)[:,i-1:i],c='red')
                                        plt.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.9)
                                else:
                                        plt.plot(day_new,timeseries_data,c='green')
                                        plt.plot(day_pred,lst_output,c='red')
                                plt.suptitle("Trained(green) vs Predicted future "+str(predict_next)+" values(blue)")
                                fig.canvas.draw()
                                img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                                img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                                img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                                pil_im=Image.fromarray(img)
                                buff=io.BytesIO()
                                pil_im.save(buff,format="PNG")
                                img_str23=base64.b64encode(buff.getvalue())
                                
                    output+="\n"
                    if ty!=3:
                            #return output,""+str(img_str2,'utf-8'),""+str(img_str,'utf-8'),"",""
                            return JsonResponse([output,""+str(img_str2,'utf-8'),""+str(img_str,'utf-8'),"",""],safe=False)
                    else:
                            if predict_future==1 and test_train==1:
                                    #return output,""+str(img_str2,'utf-8'),""+str(img_str22,'utf-8'),""+str(img_str24,'utf-8'),""+str(img_str23,'utf-8'),""
                                    return JsonResponse([output,""+str(img_str2,'utf-8'),""+str(img_str22,'utf-8'),""+str(img_str24,'utf-8'),""+str(img_str23,'utf-8'),""],safe=False)
                            elif predict_future==0 and test_train==1:
                                #return output,""+str(img_str2,'utf-8'),""+str(img_str22,'utf-8'),""+str(img_str24,'utf-8'),"",""
                                return JsonResponse([output,""+str(img_str2,'utf-8'),""+str(img_str22,'utf-8'),""+str(img_str24,'utf-8'),"",""],safe=False)
                            elif predict_future==1 and test_train==0:
                                #return output,""+str(img_str2,'utf-8'),""+str(img_str22,'utf-8'),"",""+str(img_str23,'utf-8'),""
                                return JsonResponse([output,""+str(img_str2,'utf-8'),""+str(img_str22,'utf-8'),"",""+str(img_str23,'utf-8'),""],safe=False)
                            else:
                                    #return output,""+str(img_str2,'utf-8'),""+str(img_str22,'utf-8'),"","","" 
                                    return JsonResponse([output,""+str(img_str2,'utf-8'),""+str(img_str22,'utf-8'),"","",""],safe=False)
                                
    except Exception as e:
        #return Response(e.args[0],status.HTTP_400_BAD_REQUEST)                    
        exception_type, exception_object, exception_traceback = sys.exc_info()
        line_number = exception_traceback.tb_lineno
        output="Something went wrong... Please give correct input"
        return JsonResponse(str(line_number)+str(exception_type)+" "+str(e)+"error",safe=False)
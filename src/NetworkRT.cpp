#include <iostream>
#include <list>
#include <map>
#include <errno.h>
#include <string.h> // memcpy
#include <stdlib.h>

#include "kernels.h"

#include "utils.h"
#include "NvInfer.h"
#include "NetworkRT.h"
#include "Int8Calibrator.h"

using namespace nvinfer1;

// Logger for info/warning/errors
class Logger : public ILogger {
    void log(Severity severity, const char* msg) override {
//#ifdef DEBUG
        std::cout <<"TENSORRT LOG: "<< msg << std::endl;
//#endif
    }
} loggerRT;

namespace tk { namespace dnn {

std::map<Layer*, nvinfer1::ITensor*>tensors; 

static void makeOutputMap(Network *net, std::map<int, std::list<int>>& output_map) {
	for(int i=0; i<net->num_layers; i++) {
		Layer *l = net->layers[i];
		layerType_t type = l->getLayerType();

		if(type == LAYER_SHORTCUT)  {
			Shortcut* shortcutLayer = (Shortcut *) l;
			Layer *backLayer = shortcutLayer->backLayer;

			std::map<int, std::list<int>>::iterator it;
			it = output_map.find(backLayer->id);

			if(it != output_map.end()) {
				(it->second).push_back(l->id);
			}
			else {
				std::list<int> targetLayerList;
				targetLayerList.push_back(l->id);
				output_map.insert(std::pair<int, std::list<int>>(backLayer->id, targetLayerList));
			}
		}
		else if(type == LAYER_ROUTE) {
			Route *routeLayer = (Route *) l;
			if(routeLayer->layers_n == 1 && routeLayer->groups == 1)
			{
				Layer *currLayer = routeLayer->layers[0];
				std::map<int, std::list<int>>::iterator it;
				it = output_map.find(currLayer->id);
				if(i+1 < net->num_layers)
				{
					Layer *lNext = net->layers[i+1];
					if(it != output_map.end()) {
						(it->second).push_back(lNext->id);
					}
					else {
						std::list<int> targetLayerList;
						targetLayerList.push_back(lNext->id);
						output_map.insert(std::pair<int, std::list<int>>(currLayer->id, targetLayerList));
					}		
				}
			}
			else
			{
				for(int iter = 0; iter < routeLayer->layers_n; iter++) {
					Layer *currLayer = routeLayer->layers[iter];
					std::map<int, std::list<int>>::iterator it;
					it = output_map.find(currLayer->id);
					if(it != output_map.end()) {
						(it->second).push_back(l->id);
					}
					else {
						std::list<int> targetLayerList;
						targetLayerList.push_back(l->id);
						output_map.insert(std::pair<int, std::list<int>>(currLayer->id, targetLayerList));
					}

				}
			}

		}
	}

	for(std::map<int, std::list<int>>::iterator it = output_map.begin(); it != output_map.end(); it++) {
		(it->second).sort();	
	}
}

NetworkRT::NetworkRT(Network *net, const char *name, int start_index, int end_index, int dla_core)
{
    float rt_ver = float(NV_TENSORRT_MAJOR) + 
                   float(NV_TENSORRT_MINOR)/10 + 
                   float(NV_TENSORRT_PATCH)/100;

	tensors.clear();  

	runtimeRT = nullptr;
	engineRT = nullptr;

    std::cout<<"New NetworkRT (TensorRT v"<<rt_ver<<")\n";
    builderRT = createInferBuilder(loggerRT);
    std::cout<<"Float16 support: "<<builderRT->platformHasFastFp16()<<"\n";
    std::cout<<"Int8 support: "<<builderRT->platformHasFastInt8()<<"\n";
#if NV_TENSORRT_MAJOR >= 5
    std::cout<<"DLAs: "<<builderRT->getNbDLACores()<<"\n";
#endif
    networkRT = builderRT->createNetwork();
#if NV_TENSORRT_MAJOR >= 6                
	configRT = builderRT->createBuilderConfig();
#endif

	if(net->dla) {
		is_dla = true;	
	}

    if(!fileExist(name)) {
#if NV_TENSORRT_MAJOR >= 6                
        // Calibrator life time needs to last until after the engine is built.
        std::unique_ptr<IInt8EntropyCalibrator2> calibrator;

        configRT->setAvgTimingIterations(1);
        configRT->setMinTimingIterations(1);
        configRT->setMaxWorkspaceSize(1 << 30);
        configRT->setFlag(BuilderFlag::kDEBUG);
#endif

    dataDim_t dim = net->layers[start_index]->input_dim;
    dtRT = DataType::kFLOAT;

	builderRT->setMaxBatchSize(net->maxBatchSize);
    builderRT->setMaxWorkspaceSize(1 << 30);

	if(net->fp16 && builderRT->platformHasFastFp16()) {
			dtRT = DataType::kHALF;
			builderRT->setHalf2Mode(true);
#if NV_TENSORRT_MAJOR >= 6                
			configRT->setFlag(BuilderFlag::kFP16);
#endif
	}
#if NV_TENSORRT_MAJOR >= 5
	if(net->dla && builderRT->getNbDLACores() > 0) {
			dtRT = DataType::kHALF;
			configRT->setFlag(BuilderFlag::kFP16);
			configRT->setDefaultDeviceType(DeviceType::kDLA);
            configRT->setDLACore(dla_core);
			configRT->setFlag(BuilderFlag::kGPU_FALLBACK);
			configRT->setFlag(BuilderFlag::kSTRICT_TYPES);
			is_dla = true;
	}
#endif
#if NV_TENSORRT_MAJOR >= 6                
        if(net->int8 && builderRT->platformHasFastInt8()){
            configRT->setFlag(BuilderFlag::kINT8);
            BatchStream calibrationStream(dim, 1, net->num_calib_images, net->fileImgList, net->fileLabelList);
            
            /* The calibTableFilePath contains the path+filename of the calibration table.
             * Each calibration table can be found in the corresponding network folder (../Test/*).
             * Each network is located in a folder with the same name as the network.
             * If the folder has a different name, the calibration table is saved in build/ folder.
             */
            std::string calib_table_name = std::string(name);
			calib_table_name = calib_table_name.substr(0, calib_table_name.rfind('.')) + "-calibration.table";

            calibrator.reset(new Int8EntropyCalibrator(calibrationStream, 1, 
                                            calib_table_name, 
                                            "data"));
            configRT->setInt8Calibrator(calibrator.get());
        }
#endif


	ITensor *input, *input_network;
	bool duplicated_input_flag = false;

	std::map<int, std::list<int>> output_map;
	makeOutputMap(net, output_map);

	//add other layers
	for(int i=0; i<net->num_layers; i++) {
			Layer *l = net->layers[i];
			layerType_t type = l->getLayerType();

			if(i < start_index) {
				std::map<int, std::list<int>>::iterator it;
				it = output_map.find(l->id);
				if(it != output_map.end()) {
					std::list<int>::iterator it2;
					for(it2 = (it->second).begin(); it2 != (it->second).end(); it2++) {
						Layer *tl = net->layers[(*it2)];

						if(tl->id >= start_index && tl->id <= end_index ) {
							ITensor *input_middle;
							dataDim_t outdim = l->output_dim;

							if(l->id == start_index - 1) {
								if(!duplicated_input_flag) {
									if(start_index > 0 )
 										input_middle = networkRT->addInput("data", DataType::kHALF, DimsCHW{ outdim.c, outdim.h, outdim.w });
									else
										input_middle = networkRT->addInput("data", DataType::kFLOAT, DimsCHW{ outdim.c, outdim.h, outdim.w });

									checkNULL(input_middle);
									tensors[l] = input_middle;
									duplicated_input_flag = true;
									input_network = input_middle;
								}
							}
							else {

								if(tensors.find(l) == tensors.end())
								{
									input_middle = networkRT->addInput((l->getLayerName() + std::to_string(l->id) + "To" + std::to_string(tl->id) + "_out").c_str(), DataType::kHALF, DimsCHW{ outdim.c, outdim.h, outdim.w });
									checkNULL(input_middle);
									tensors[l] = input_middle;
								}
							}
						}
					}
				}

				continue;
			}
			else if(i == start_index) {
				if(i > 0)
				{
					Layer *lBefore = net->layers[i-1];
					if(lBefore->getLayerType() == LAYER_ROUTE && ((Route *)lBefore)->layers_n == 1 && ((Route *)lBefore)->groups == 1)
					{
						input = tensors[((Route *)lBefore)->layers[0]];
					}
					else
					{
						if(!duplicated_input_flag) {
							if(start_index > 0)
								input = networkRT->addInput("data", DataType::kHALF, DimsCHW{ dim.c, dim.h, dim.w });
							else
								input = networkRT->addInput("data", DataType::kFLOAT, DimsCHW{ dim.c, dim.h, dim.w });
							checkNULL(input);
						}
						else {
							input = input_network;
						}
					}
				}
				else
				{
					if(start_index > 0)
						input = networkRT->addInput("data", DataType::kHALF, DimsCHW{ dim.c, dim.h, dim.w });
					else
						input = networkRT->addInput("data", DataType::kFLOAT, DimsCHW{ dim.c, dim.h, dim.w });
					checkNULL(input);			
				}
			}

			if(l->getLayerType() == LAYER_ROUTE && ((Route *)l)->layers_n == 1 && ((Route *)l)->groups == 1) {
				input = tensors[((Route *)l)->layers[0]];
			}
			else {
				ILayer *Ilay = convert_layer(input, l);
#if NV_TENSORRT_MAJOR >= 6                
				if(net->int8 && builderRT->platformHasFastInt8())
				{
					Ilay->setPrecision(DataType::kINT8);
				}
#endif
				if (configRT->canRunOnDLA(Ilay) == true && net->dla == true) {
					std::cout << "DLA layer: " << i << ", name: " << l->getLayerName() << std::endl;
					configRT->setDeviceType(Ilay, DeviceType::kDLA);
				}
				else
				{
					std::cout << "GPU layer: " << i << ", name: " << l->getLayerName() << std::endl;
				}
				Ilay->setName( (l->getLayerName() + std::to_string(i)).c_str() );

				input = Ilay->getOutput(0);
				input->setName( (l->getLayerName() + std::to_string(i) + "_out").c_str() );
			}

			if(l->final) {
				networkRT->markOutput(*input);
			}
			tensors[l] = input;

			if(end_index == i) {
				break;
			}
	}
	
	for(int i=end_index+1; i<net->num_layers; i++) 	
	{
		Layer *l = net->layers[i];
		layerType_t type = l->getLayerType();
		if(type == LAYER_SHORTCUT) 
		{
			Shortcut* shortcutLayer = (Shortcut *) l;
			std::map<Layer*, nvinfer1::ITensor*>::iterator it = tensors.find(shortcutLayer->backLayer);
			if(it != tensors.end()) 
			{
				//if(shortcutLayer->backLayer->output_dim.c != shortcutLayer->output_dim.c) FatalError("Different shortcut size for output is not supported.");
				it->second->setType(DataType::kHALF);
				networkRT->markOutput(*(it->second));
			}
		}
		else if(type == LAYER_ROUTE)
		{
			Route *routeLayer = (Route *) l;
			for(int iter = 0; iter < routeLayer->layers_n; iter++) {
				Layer *currLayer = routeLayer->layers[iter];
				std::map<Layer*, nvinfer1::ITensor*>::iterator it = tensors.find(currLayer);
				if(it != tensors.end())  {
					it->second->setType(DataType::kHALF);
					networkRT->markOutput(*(it->second));
				}
			}
		}
	}

	//build tensorRT
	input->setName("out");
	if(end_index + 1 < net->num_layers)
	{
		input->setType(DataType::kHALF);
	}

	networkRT->markOutput(*input);

	if(input == NULL)
			FatalError("conversion failed");


	std::cout<<"Selected maxBatchSize: "<<builderRT->getMaxBatchSize()<<"\n";
	printCudaMemUsage();
	std::cout<<"Building tensorRT cuda engine...\n";
#if NV_TENSORRT_MAJOR >= 6                
	engineRT = builderRT->buildEngineWithConfig(*networkRT, *configRT);
#else 
	engineRT = builderRT->buildCudaEngine(*networkRT);
#endif
    if(engineRT == nullptr)
        FatalError("cloud not build cuda engine")
    std::cout<<"serialize net\n";
    serialize(name);
	networkRT->destroy();
	builderRT->destroy();
	configRT->destroy();
	output_map.clear();

    } 
    deserialize(name, dla_core);

	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	std::cout<<"Input/outputs numbers: "<<engineRT->getNbBindings()<<"\n";
    if(engineRT->getNbBindings() > MAX_BUFFERS_RT)
        FatalError("over RT buffer array size");
}

NetworkRT::NetworkRT(Network *net, const char *name) {

    float rt_ver = float(NV_TENSORRT_MAJOR) + 
                   float(NV_TENSORRT_MINOR)/10 + 
                   float(NV_TENSORRT_PATCH)/100;
    std::cout<<"New NetworkRT (TensorRT v"<<rt_ver<<")\n";
  
    builderRT = createInferBuilder(loggerRT);
    std::cout<<"Float16 support: "<<builderRT->platformHasFastFp16()<<"\n";
    std::cout<<"Int8 support: "<<builderRT->platformHasFastInt8()<<"\n";
#if NV_TENSORRT_MAJOR >= 5
    std::cout<<"DLAs: "<<builderRT->getNbDLACores()<<"\n";
#endif
    networkRT = builderRT->createNetwork();
#if NV_TENSORRT_MAJOR >= 6                
        configRT = builderRT->createBuilderConfig();
#endif
    
    if(!fileExist(name)) {
#if NV_TENSORRT_MAJOR >= 6                
        // Calibrator life time needs to last until after the engine is built.
        std::unique_ptr<IInt8EntropyCalibrator2> calibrator;

        configRT->setAvgTimingIterations(1);
        configRT->setMinTimingIterations(1);
        configRT->setMaxWorkspaceSize(1 << 30);
        configRT->setFlag(BuilderFlag::kDEBUG);
#endif
        //input and datatype
        dataDim_t dim = net->layers[0]->input_dim;
        dtRT = DataType::kFLOAT;

        builderRT->setMaxBatchSize(net->maxBatchSize);
        builderRT->setMaxWorkspaceSize(1 << 30);

        if(net->fp16 && builderRT->platformHasFastFp16()) {
            dtRT = DataType::kHALF;
            builderRT->setHalf2Mode(true);
#if NV_TENSORRT_MAJOR >= 6                
            configRT->setFlag(BuilderFlag::kFP16);
#endif
        }
#if NV_TENSORRT_MAJOR >= 5
        if(net->dla && builderRT->getNbDLACores() > 0) {
            dtRT = DataType::kHALF;
            //builderRT->setfp16mode(true);
            //builderRT->allowGPUFallback(true);
            //builderRT->setDefaultDeviceType(DeviceType::kDLA);
            //builderRT->setDLACore(0);
            configRT->setFlag(BuilderFlag::kFP16);
            configRT->setDefaultDeviceType(DeviceType::kDLA);
            configRT->setDLACore(0);
			configRT->setFlag(BuilderFlag::kGPU_FALLBACK);
        }
#endif
#if NV_TENSORRT_MAJOR >= 6                
        if(net->int8 && builderRT->platformHasFastInt8()){
            // dtRT = DataType::kINT8;
            // builderRT->setInt8Mode(true);
            configRT->setFlag(BuilderFlag::kINT8);
            BatchStream calibrationStream(dim, 1, 100,      //TODO: check if 100 images are sufficient to the calibration (or 4951) 
                                            net->fileImgList, net->fileLabelList);
            
            /* The calibTableFilePath contains the path+filename of the calibration table.
             * Each calibration table can be found in the corresponding network folder (../Test/*).
             * Each network is located in a folder with the same name as the network.
             * If the folder has a different name, the calibration table is saved in build/ folder.
             */
            std::string calib_table_name = net->networkName + "/" + net->networkNameRT.substr(0, net->networkNameRT.find('.')) + "-calibration.table";
            std::string calib_table_path = net->networkName;
            if(!fileExist((const char *)calib_table_path.c_str()))
                calib_table_name = "./" + net->networkNameRT.substr(0, net->networkNameRT.find('.')) + "-calibration.table";

            calibrator.reset(new Int8EntropyCalibrator(calibrationStream, 1, 
                                            calib_table_name, 
                                            "data"));
            configRT->setInt8Calibrator(calibrator.get());
        }
#endif
        
        // add input layer
        ITensor *input = networkRT->addInput("data", DataType::kFLOAT, 
                        DimsCHW{ dim.c, dim.h, dim.w});
        checkNULL(input);

        //add other layers
        for(int i=0; i<net->num_layers; i++) {
            Layer *l = net->layers[i];
            ILayer *Ilay = convert_layer(input, l);
#if NV_TENSORRT_MAJOR >= 6                
            if(net->int8 && builderRT->platformHasFastInt8())
            {
                Ilay->setPrecision(DataType::kINT8);
            }
#endif
            Ilay->setName( (l->getLayerName() + std::to_string(i)).c_str() );
            
            input = Ilay->getOutput(0);
            input->setName( (l->getLayerName() + std::to_string(i) + "_out").c_str() );
            
            if(l->final)
                networkRT->markOutput(*input);
            tensors[l] = input;
        }
        if(input == NULL)
            FatalError("conversion failed");

        //build tensorRT
        input->setName("out");
        networkRT->markOutput(*input);

        std::cout<<"Selected maxBatchSize: "<<builderRT->getMaxBatchSize()<<"\n";
        printCudaMemUsage();
        std::cout<<"Building tensorRT cuda engine...\n";
#if NV_TENSORRT_MAJOR >= 6                
        engineRT = builderRT->buildEngineWithConfig(*networkRT, *configRT);
#else 
        engineRT = builderRT->buildCudaEngine(*networkRT);
#endif
        if(engineRT == nullptr)
            FatalError("cloud not build cuda engine")
        // we don't need the network any more
        //networkRT->destroy();
        std::cout<<"serialize net\n";
        serialize(name);
    }
    deserialize(name);

    std::cout<<"create execution context\n";
	contextRT = engineRT->createExecutionContext();

	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	std::cout<<"Input/outputs numbers: "<<engineRT->getNbBindings()<<"\n";
    if(engineRT->getNbBindings() > MAX_BUFFERS_RT)
        FatalError("over RT buffer array size");

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	buf_input_idx = engineRT->getBindingIndex("data"); 
    buf_output_idx = engineRT->getBindingIndex("out");
    std::cout<<"input index = "<<buf_input_idx<<" -> output index = "<<buf_output_idx<<"\n";


    Dims iDim = engineRT->getBindingDimensions(buf_input_idx);
    input_dim.n = 1;
    input_dim.c = iDim.d[0];
    input_dim.h = iDim.d[1];
    input_dim.w = iDim.d[2];
    input_dim.print();

    Dims oDim = engineRT->getBindingDimensions(buf_output_idx);
    output_dim.n = 1;
    output_dim.c = oDim.d[0];
    output_dim.h = oDim.d[1];
    output_dim.w = oDim.d[2];
    output_dim.print();
	
    // create GPU buffers and a stream
    for(int i=0; i<engineRT->getNbBindings(); i++) {
        Dims dim = engineRT->getBindingDimensions(i);
        buffersDIM[i] = dataDim_t(1, dim.d[0], dim.d[1], dim.d[2]);
        std::cout<<"RtBuffer "<<i<<"   dim: "; buffersDIM[i].print();
        checkCuda(cudaMalloc(&buffersRT[i], engineRT->getMaxBatchSize()*dim.d[0]*dim.d[1]*dim.d[2]*sizeof(dnnType)));
    }
    checkCuda(cudaMalloc(&output, engineRT->getMaxBatchSize()*output_dim.tot()*sizeof(dnnType)));
	checkCuda(cudaStreamCreate(&stream));
}

NetworkRT::~NetworkRT() {
	if(engineRT != nullptr){
		engineRT->destroy();
	}
		
	if(runtimeRT != nullptr) {
		runtimeRT->destroy();
	}

/*	if(builderRT != nullptr) {
		builderRT->destroy();
	}

	if(networkRT != nullptr) {
		networkRT->destroy();
	}

	if(configRT != nullptr) {
		configRT->destroy();
	}
*/

}

void NetworkRT::run_on_dla(ILayer*l) {
	if (is_dla == true && configRT->canRunOnDLA(l) == true) {
		configRT->setDeviceType(l, DeviceType::kDLA);
	}
}

dnnType* NetworkRT::infer(dataDim_t &dim, dnnType* data) {
    int batches = dim.n;
    if(batches > getMaxBatchSize()) {
        FatalError("input batch size too large");
    }

    checkCuda(cudaMemcpyAsync(buffersRT[buf_input_idx], data, batches*input_dim.tot()*sizeof(dnnType), cudaMemcpyDeviceToDevice, stream));
    contextRT->enqueue(batches, buffersRT, stream, nullptr);
    checkCuda(cudaMemcpyAsync(output, buffersRT[buf_output_idx], batches*output_dim.tot()*sizeof(dnnType), cudaMemcpyDeviceToDevice, stream));
    checkCuda(cudaStreamSynchronize(stream));

    dim = output_dim;
    dim.n = batches;

    return output;
}

void NetworkRT::enqueue(int batchSize) {
    contextRT->enqueue(batchSize, buffersRT, stream, nullptr);
}

ILayer* NetworkRT::convert_layer(ITensor *input, Layer *l) {

    layerType_t type = l->getLayerType();

    if(type == LAYER_DENSE)
        return convert_layer(input, (Dense*) l);
    if(type == LAYER_CONV2D || type == LAYER_DECONV2D)
        return convert_layer(input, (Conv2d*) l);
    if(type == LAYER_POOLING)
        return convert_layer(input, (Pooling*) l);
    if(type == LAYER_ACTIVATION || type == LAYER_ACTIVATION_CRELU || type == LAYER_ACTIVATION_LEAKY || type == LAYER_ACTIVATION_MISH || type == LAYER_ACTIVATION_LOGISTIC)
        return convert_layer(input, (Activation*) l);
    if(type == LAYER_SOFTMAX)
        return convert_layer(input, (Softmax*) l);
    if(type == LAYER_ROUTE)
        return convert_layer(input, (Route*) l);
    if(type == LAYER_FLATTEN)
        return convert_layer(input, (Flatten*) l);
    if(type == LAYER_RESHAPE)
        return convert_layer(input, (Reshape*) l);
    if(type == LAYER_REORG)
        return convert_layer(input, (Reorg*) l);
    if(type == LAYER_REGION)
        return convert_layer(input, (Region*) l);
    if(type == LAYER_SHORTCUT)
        return convert_layer(input, (Shortcut*) l);
    if(type == LAYER_YOLO)
        return convert_layer(input, (Yolo*) l);
    if(type == LAYER_UPSAMPLE)
        return convert_layer(input, (Upsample*) l);
    if(type == LAYER_DEFORMCONV2D)
        return convert_layer(input, (DeformConv2d*) l);

    std::cout<<l->getLayerName()<<"\n";
    FatalError("Layer not implemented in tensorRT");
    return NULL;
}

ILayer* NetworkRT::convert_layer(ITensor *input, Dense *l) {
    //std::cout<<"convert Dense\n";
    void *data_b, *bias_b;
    if(dtRT == DataType::kHALF) {
        data_b     = l->data16_h;    
        bias_b     = l->bias16_h;
    } else {
        data_b     = l->data_h;    
        bias_b     = l->bias_h;
    }

    Weights w { dtRT, data_b, l->inputs*l->outputs};
    Weights b = { dtRT, bias_b, l->outputs};
    IFullyConnectedLayer *lRT = networkRT->addFullyConnected(*input, l->outputs, w, b);

    checkNULL(lRT);
    return lRT;
}


ILayer* NetworkRT::convert_layer(ITensor *input, Conv2d *l) {
    // std::cout<<"convert conv2D\n";
    // printf("%d %d %d %d %d\n", l->kernelH, l->kernelW, l->inputs, l->outputs, l->batchnorm);


    void *data_b, *bias_b, *bias2_b, *power_b, *mean_b, *variance_b, *scales_b;
    if(dtRT == DataType::kHALF) {
        data_b     = l->data16_h;    
        bias_b     = l->bias16_h;
        bias2_b    = l->bias216_h;
        power_b    = l->power16_h;
        mean_b     = l->mean16_h;
        variance_b = l->variance16_h;
        scales_b   = l->scales16_h;
    } else {
        data_b     = l->data_h;    
        bias_b     = l->bias_h;
        bias2_b    = l->bias2_h;
        power_b    = l->power_h;
        mean_b     = l->mean_h;
        variance_b = l->variance_h;
        scales_b   = l->scales_h;
    }


    Weights w { dtRT, data_b, l->inputs*l->outputs*l->kernelH*l->kernelW};
    Weights b;
    if(!l->batchnorm)
        b = { dtRT, bias_b, l->outputs};
    else{
        if (l->additional_bias)
            b = { dtRT, bias2_b, l->outputs}; 
        else
            b = { dtRT, nullptr, 0}; //on batchnorm bias are added later
    }

    ILayer *lRT = nullptr;
    if(!l->deConv) {
        IConvolutionLayer *lRTconv = networkRT->addConvolution(*input, 
            l->outputs, DimsHW{l->kernelH, l->kernelW}, w, b);
        checkNULL(lRTconv);
        lRTconv->setStride(DimsHW{l->strideH, l->strideW});
        lRTconv->setPadding(DimsHW{l->paddingH, l->paddingW});
        lRTconv->setNbGroups(l->groups);
        lRT = (ILayer*) lRTconv;
    } else {
        IDeconvolutionLayer *lRTconv = networkRT->addDeconvolution(*input, 
            l->outputs, DimsHW{l->kernelH, l->kernelW}, w, b);
        checkNULL(lRTconv);
        lRTconv->setStride(DimsHW{l->strideH, l->strideW});
        lRTconv->setPadding(DimsHW{l->paddingH, l->paddingW});
        lRTconv->setNbGroups(l->groups);
        lRT = (ILayer*) lRTconv;
        
        Dims d = lRTconv->getOutput(0)->getDimensions();
        //std::cout<<"DECONV: "<<d.d[0]<<" "<<d.d[1]<<" "<<d.d[2]<<" "<<d.d[3]<<"\n";
    }

    checkNULL(lRT);
	run_on_dla(lRT);

    if(l->batchnorm) {
        Weights power{dtRT, power_b, l->outputs};
        Weights shift{dtRT, mean_b, l->outputs};
        Weights scale{dtRT, variance_b, l->outputs};
        // std::cout<<lRT->getNbOutputs()<<std::endl;
        IScaleLayer *lRT2 = networkRT->addScale(*lRT->getOutput(0), ScaleMode::kCHANNEL, 
                    shift, scale, power);
        
        checkNULL(lRT2);
		run_on_dla(lRT2);

        Weights shift2{dtRT, bias_b, l->outputs};
        Weights scale2{dtRT, scales_b, l->outputs};
        IScaleLayer *lRT3 = networkRT->addScale(*lRT2->getOutput(0), ScaleMode::kCHANNEL, 
                    shift2, scale2, power);
        checkNULL(lRT3);
		run_on_dla(lRT3);

        return lRT3;
    }

    return lRT;
}

ILayer* NetworkRT::convert_layer(ITensor *input, Pooling *l) {
    // std::cout<<"convert Pooling\n";

    PoolingType ptype;
    if(l->pool_mode == tkdnnPoolingMode_t::POOLING_MAX) ptype = PoolingType::kMAX;
    if(l->pool_mode == tkdnnPoolingMode_t::POOLING_AVERAGE) ptype = PoolingType::kAVERAGE;
    if(l->pool_mode == tkdnnPoolingMode_t::POOLING_AVERAGE_EXCLUDE_PADDING) ptype = PoolingType::kMAX_AVERAGE_BLEND;

    if(l->pool_mode == tkdnnPoolingMode_t::POOLING_MAX_FIXEDSIZE)
    {
        IPlugin *plugin = new MaxPoolFixedSizeRT(l->output_dim.c, l->output_dim.h, l->output_dim.w, l->output_dim.n, l->strideH, l->strideW, l->winH, l->winH-1);        
        IPluginLayer *lRT = networkRT->addPlugin(&input, 1, *plugin);
        checkNULL(lRT);
        return lRT;
    }
    else
    {
        IPoolingLayer *lRT = networkRT->addPooling(*input, ptype, DimsHW{l->winH, l->winW});
        checkNULL(lRT);

        lRT->setPadding(DimsHW{l->paddingH, l->paddingW});
        lRT->setStride(DimsHW{l->strideH, l->strideW});
        return lRT;
    }  
}

ILayer* NetworkRT::convert_layer(ITensor *input, Activation *l) {
    //std::cout<<"convert Activation\n";

    if(l->act_mode == ACTIVATION_LEAKY) {
        //std::cout<<"New plugin LEAKY\n";
        
#if NV_TENSORRT_MAJOR < 6                
        // plugin version
        IPlugin *plugin = new ActivationLeakyRT();
        IPluginLayer *lRT = networkRT->addPlugin(&input, 1, *plugin);
        checkNULL(lRT);
        return lRT;
#else 
        IActivationLayer *lRT = networkRT->addActivation(*input, ActivationType::kLEAKY_RELU);
        lRT->setAlpha(0.1);
        checkNULL(lRT);
        return lRT;
#endif

    } else if(l->act_mode == CUDNN_ACTIVATION_RELU) {
        IActivationLayer *lRT = networkRT->addActivation(*input, ActivationType::kRELU);
        checkNULL(lRT);
        return lRT;
    } else if(l->act_mode == CUDNN_ACTIVATION_SIGMOID) {
        IActivationLayer *lRT = networkRT->addActivation(*input, ActivationType::kSIGMOID);
        checkNULL(lRT);
        return lRT;
    }
    else if(l->act_mode == CUDNN_ACTIVATION_CLIPPED_RELU) {
        IPlugin *plugin = new ActivationReLUCeiling(l->ceiling);
        IPluginLayer *lRT = networkRT->addPlugin(&input, 1, *plugin);
        checkNULL(lRT);
        return lRT;
    } 
    else if(l->act_mode == ACTIVATION_MISH) {
        IPlugin *plugin = new ActivationMishRT();
        IPluginLayer *lRT = networkRT->addPlugin(&input, 1, *plugin);
        checkNULL(lRT);
        return lRT;
    }
    else if(l->act_mode == ACTIVATION_LOGISTIC) {
        IPlugin *plugin = new ActivationLogisticRT();
        IPluginLayer *lRT = networkRT->addPlugin(&input, 1, *plugin);
        checkNULL(lRT);
        return lRT;
    }
    else {
        FatalError("this Activation mode is not yet implemented");
        return NULL;
    }
}

ILayer* NetworkRT::convert_layer(ITensor *input, Softmax *l) {
    //std::cout<<"convert softmax\n";

    ISoftMaxLayer *lRT = networkRT->addSoftMax(*input);
    checkNULL(lRT);

    return lRT;
}

ILayer* NetworkRT::convert_layer(ITensor *input, Route *l) {
    // std::cout<<"convert route\n";

    ITensor **tens = new ITensor*[l->layers_n];
    for(int i=0; i<l->layers_n; i++) {
        tens[i] = tensors[l->layers[i]];

		/*
        for(int j=0; j<tens[i]->getDimensions().nbDims; j++) {
            std::cout<<tens[i]->getDimensions().d[j]<<" ";
        }
        std::cout<<"\n";
		*/
    }

    if(l->groups > 1){
        IPluginExt *plugin = new RouteRT(l->groups, l->group_id);
        IPluginLayer *lRT = networkRT->addPluginExt(tens, l->layers_n, *plugin);
		
        checkNULL(lRT);
        return lRT;
    }

    if(l->layers_n > 1 && is_dla == true)
    {
        for(int i=0; i<l->layers_n; i++) {
            ITensor *back_tens = tensors[l->layers[i]];
            IPoolingLayer *lPool = networkRT->addPooling(*back_tens, PoolingType::kMAX, DimsHW{1, 1});
            lPool->setStride(DimsHW{1, 1});
            run_on_dla(lPool);
            tens[i] = lPool->getOutput(0);
        }  
    }   

    IConcatenationLayer *lRT = networkRT->addConcatenation(tens, l->layers_n);
    checkNULL(lRT);
    return lRT;
}

ILayer* NetworkRT::convert_layer(ITensor *input, Flatten *l) {

    IPlugin *plugin = new FlattenConcatRT();
    IPluginLayer *lRT = networkRT->addPlugin(&input, 1, *plugin);
    checkNULL(lRT);
    return lRT;
}

ILayer* NetworkRT::convert_layer(ITensor *input, Reshape *l) {
    // std::cout<<"convert Reshape\n";

    l->output_dim.print();
    IPlugin *plugin = new ReshapeRT(l->output_dim);
    IPluginLayer *lRT = networkRT->addPlugin(&input, 1, *plugin);
    checkNULL(lRT);
    return lRT;
}

ILayer* NetworkRT::convert_layer(ITensor *input, Reorg *l) {
    //std::cout<<"convert Reorg\n";

    //std::cout<<"New plugin REORG\n";
    IPlugin *plugin = new ReorgRT(l->stride);
    IPluginLayer *lRT = networkRT->addPlugin(&input, 1, *plugin);
    checkNULL(lRT);
    return lRT;
}

ILayer* NetworkRT::convert_layer(ITensor *input, Region *l) {
    //std::cout<<"convert Region\n";

    //std::cout<<"New plugin REGION\n";
    IPlugin *plugin = new RegionRT(l->classes, l->coords, l->num);
    IPluginLayer *lRT = networkRT->addPlugin(&input, 1, *plugin);
    checkNULL(lRT);
    return lRT;
}

ILayer* NetworkRT::convert_layer(ITensor *input, Shortcut *l) {
    //std::cout<<"convert Shortcut\n";

    //std::cout<<"New plugin Shortcut\n";
    
    ITensor *back_tens = tensors[l->backLayer];

    if(l->backLayer->output_dim.c == l->output_dim.c)
    {
		if(is_dla == true)
		{
			IPoolingLayer *lPool = networkRT->addPooling(*back_tens, PoolingType::kMAX, DimsHW{1, 1});
    	    lPool->setStride(DimsHW{1, 1});
	        run_on_dla(lPool);
    	    back_tens = lPool->getOutput(0);
		}

        IElementWiseLayer *lRT = networkRT->addElementWise(*back_tens, *input, ElementWiseOperation::kSUM);
        checkNULL(lRT);
        return lRT;
    }
    else
    {
        // plugin version
        IPlugin *plugin = new ShortcutRT(l->backLayer->output_dim);
        ITensor **inputs = new ITensor*[2];
        inputs[0] = input;
        inputs[1] = back_tens; 
        IPluginLayer *lRT = networkRT->addPlugin(inputs, 2, *plugin);
        checkNULL(lRT);
        return lRT;
    }
}

ILayer* NetworkRT::convert_layer(ITensor *input, Yolo *l) {
    //std::cout<<"convert Yolo\n";

    //std::cout<<"New plugin YOLO\n";
    IPlugin *plugin = new YoloRT(l->classes, l->num, l, l->n_masks, l->scaleXY, l->nms_thresh, l->nsm_kind, l->new_coords);
    IPluginLayer *lRT = networkRT->addPlugin(&input, 1, *plugin);
    checkNULL(lRT);
    return lRT;
}

ILayer* NetworkRT::convert_layer(ITensor *input, Upsample *l) {
    //std::cout<<"convert Upsample\n";

    //std::cout<<"New plugin UPSAMPLE\n";
    IPlugin *plugin = new UpsampleRT(l->stride);
    IPluginLayer *lRT = networkRT->addPlugin(&input, 1, *plugin);
    checkNULL(lRT);
    return lRT;
}

ILayer* NetworkRT::convert_layer(ITensor *input, DeformConv2d *l) {
    //std::cout<<"convert DEFORMABLE\n";
    ILayer *preconv = convert_layer(input, l->preconv);
    checkNULL(preconv);

    ITensor **inputs = new ITensor*[2];
    inputs[0] = input;
    inputs[1] = preconv->getOutput(0);

    //std::cout<<"New plugin DEFORMABLE\n";
    IPlugin *plugin = new DeformableConvRT(l->chunk_dim, l->kernelH, l->kernelW, l->strideH, l->strideW, l->paddingH, l->paddingW, 
                                            l->deformableGroup, l->input_dim.n, l->input_dim.c, l->input_dim.h, l->input_dim.w, 
                                            l->output_dim.n, l->output_dim.c, l->output_dim.h, l->output_dim.w, l);
    IPluginLayer *lRT = networkRT->addPlugin(inputs, 2, *plugin);
    checkNULL(lRT);
    lRT->setName( ("Deformable" + std::to_string(l->id)).c_str() );
	run_on_dla(lRT);
    delete(inputs);
    // batchnorm
    void *bias_b, *power_b, *mean_b, *variance_b, *scales_b;
    if(dtRT == DataType::kHALF) {
        bias_b     = l->bias16_h;
        power_b    = l->power16_h;
        mean_b     = l->mean16_h;
        variance_b = l->variance16_h;
        scales_b   = l->scales16_h;
    } else {
        bias_b     = l->bias_h;
        power_b    = l->power_h;
        mean_b     = l->mean_h;
        variance_b = l->variance_h;
        scales_b   = l->scales_h;
    }

    Weights power{dtRT, power_b, l->outputs};
    Weights shift{dtRT, mean_b, l->outputs};
    Weights scale{dtRT, variance_b, l->outputs};
    //std::cout<<lRT->getNbOutputs()<<std::endl;
    IScaleLayer *lRT2 = networkRT->addScale(*lRT->getOutput(0), ScaleMode::kCHANNEL, 
                shift, scale, power);
    
    checkNULL(lRT2);
	run_on_dla(lRT2);

    Weights shift2{dtRT, bias_b, l->outputs};
    Weights scale2{dtRT, scales_b, l->outputs};
    IScaleLayer *lRT3 = networkRT->addScale(*lRT2->getOutput(0), ScaleMode::kCHANNEL, 
                shift2, scale2, power);
    checkNULL(lRT3);
	run_on_dla(lRT3);

    return lRT3;
}

bool NetworkRT::serialize(const char *filename) {

    std::ofstream p(filename, std::ios::binary);
    if (!p) {
        FatalError("could not open plan output file");
        return false;
    }

    IHostMemory *ptr = engineRT->serialize();
    if(ptr == nullptr)
        FatalError("Cant serialize network");

    p.write(reinterpret_cast<const char*>(ptr->data()), ptr->size());
    ptr->destroy();
	engineRT->destroy();
    engineRT = nullptr;

    return true;
}

bool NetworkRT::deserialize(const char *filename) {

    char *gieModelStream{nullptr};
    size_t size{0};
    std::ifstream file(filename, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        gieModelStream = new char[size];
        file.read(gieModelStream, size);
        file.close();
    }

    pluginFactory = new PluginFactory();
    runtimeRT = createInferRuntime(loggerRT);
    engineRT = runtimeRT->deserializeCudaEngine(gieModelStream, size, (IPluginFactory *) pluginFactory);
    if (gieModelStream) delete [] gieModelStream;

    return true;
}

bool NetworkRT::deserialize(const char *filename, int dla_core) {

    char *gieModelStream{nullptr};
    size_t size{0};
    std::ifstream file(filename, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        gieModelStream = new char[size];
        file.read(gieModelStream, size);
        file.close();
    }

    pluginFactory = new PluginFactory();
    runtimeRT = createInferRuntime(loggerRT);
	if(is_dla) {
		runtimeRT->setDLACore(dla_core);
	}
    engineRT = runtimeRT->deserializeCudaEngine(gieModelStream, size, (IPluginFactory *) pluginFactory);
    if (gieModelStream) delete [] gieModelStream;

    return true;
}

IPlugin* PluginFactory::createPlugin(const char* layerName, const void* serialData, size_t serialLength) {
    const char * buf = reinterpret_cast<const char*>(serialData);

    std::string name(layerName);
    //std::cout<<name<<std::endl;

    if(name.find("ActivationLeaky") == 0) {
        ActivationLeakyRT *a = new ActivationLeakyRT();
        a->size = readBUF<int>(buf);
        return a;
    }
    if(name.find("ActivationMish") == 0) {
        ActivationMishRT *a = new ActivationMishRT();
        a->size = readBUF<int>(buf);
        return a;
    }
    if(name.find("ActivationLogistic") == 0) {
        ActivationLogisticRT *a = new ActivationLogisticRT();
        a->size = readBUF<int>(buf);
        return a;
    }
    if(name.find("ActivationCReLU") == 0) {
        ActivationReLUCeiling *a = new ActivationReLUCeiling(readBUF<float>(buf));
        a->size = readBUF<int>(buf);
        return a;
    }

    if(name.find("Region") == 0) {
        RegionRT *r = new RegionRT(readBUF<int>(buf),    //classes
                                    readBUF<int>(buf),    //coords
                                    readBUF<int>(buf));   //num

        r->c = readBUF<int>(buf);
        r->h = readBUF<int>(buf);
        r->w = readBUF<int>(buf);
        return r;
    } 

    if(name.find("Reorg") == 0) {
        ReorgRT *r = new ReorgRT(readBUF<int>(buf)); //stride
        r->c = readBUF<int>(buf);
        r->h = readBUF<int>(buf);
        r->w = readBUF<int>(buf);
        return r;
    } 

    if(name.find("Shortcut") == 0) {
        tk::dnn::dataDim_t bdim;
        bdim.c = readBUF<int>(buf);
        bdim.h = readBUF<int>(buf);
        bdim.w = readBUF<int>(buf);
        bdim.l = 1;

        ShortcutRT *r = new ShortcutRT(bdim);
        r->c = readBUF<int>(buf);
        r->h = readBUF<int>(buf);
        r->w = readBUF<int>(buf);
        return r;
    } 

    if(name.find("Pooling") == 0) {
        MaxPoolFixedSizeRT *r = new MaxPoolFixedSizeRT( readBUF<int>(buf), //c
                                                        readBUF<int>(buf), //h
                                                        readBUF<int>(buf), //w
                                                        readBUF<int>(buf), //n
                                                        readBUF<int>(buf), //strideH
                                                        readBUF<int>(buf), //strideW
                                                        readBUF<int>(buf), //winSize
                                                        readBUF<int>(buf)); //padding
        return r;
    }

    if(name.find("Resize") == 0) {
        ResizeLayerRT *r = new ResizeLayerRT(readBUF<int>(buf), //o_c
                                            readBUF<int>(buf), //o_h
                                            readBUF<int>(buf)); //o_w
        r->i_c = readBUF<int>(buf);
        r->i_h = readBUF<int>(buf);
        r->i_w = readBUF<int>(buf);
        return r;
    } 

    if(name.find("Flatten") == 0) {
        FlattenConcatRT *r = new FlattenConcatRT(); 
        r->c = readBUF<int>(buf);
        r->h = readBUF<int>(buf);
        r->w = readBUF<int>(buf);
        r->rows = readBUF<int>(buf);
        r->cols = readBUF<int>(buf);
        return r;
    } 

    if(name.find("Reshape") == 0) {

        dataDim_t new_dim;
        new_dim.n = readBUF<int>(buf);
        new_dim.c = readBUF<int>(buf);
        new_dim.h = readBUF<int>(buf);
        new_dim.w = readBUF<int>(buf);
        ReshapeRT *r = new ReshapeRT(new_dim); 
        
        return r;
    } 

    if(name.find("Yolo") == 0) {
        YoloRT *r = new YoloRT(readBUF<int>(buf),    //classes
                                readBUF<int>(buf),   //num
                                nullptr, //yolo
                                readBUF<int>(buf), //n_masks
                                readBUF<float>(buf), //scale_xy
                                readBUF<float>(buf),  //nms_thresh 
                                readBUF<int>(buf),  //nms_kind
                                readBUF<int>(buf)  //new_coords
                                );   
        r->c = readBUF<int>(buf);
        r->h = readBUF<int>(buf);
        r->w = readBUF<int>(buf);
        for(int i=0; i<r->n_masks; i++)
            r->mask[i] = readBUF<dnnType>(buf);
        for(int i=0; i<r->n_masks*2*r->num; i++)
            r->bias[i] = readBUF<dnnType>(buf);

		// save classes names
        r->classesNames.resize(r->classes);
		for(int i=0; i<r->classes; i++) {
            char tmp[YOLORT_CLASSNAME_W];
			for(int j=0; j<YOLORT_CLASSNAME_W; j++)
				tmp[j] = readBUF<char>(buf);
            r->classesNames[i] = std::string(tmp);
		}

        yolos[n_yolos++] = r;
        return r;
    } 
    if(name.find("Upsample") == 0) {
        UpsampleRT *r = new UpsampleRT(readBUF<int>(buf)); //stride
        r->c = readBUF<int>(buf);
        r->h = readBUF<int>(buf);
        r->w = readBUF<int>(buf);
        return r;
    } 

    if(name.find("Route") == 0) {
        RouteRT *r = new RouteRT(readBUF<int>(buf),readBUF<int>(buf));
        r->in = readBUF<int>(buf);
        for(int i=0; i<RouteRT::MAX_INPUTS; i++)
            r->c_in[i] = readBUF<int>(buf);
        r->c = readBUF<int>(buf);
        r->h = readBUF<int>(buf);
        r->w = readBUF<int>(buf);
		r->mDataType = readBUF<nvinfer1::DataType>(buf);
        return r;
    } 

    if(name.find("Deformable") == 0) {
        DeformableConvRT *r = new DeformableConvRT(readBUF<int>(buf), readBUF<int>(buf), readBUF<int>(buf),
                                                    readBUF<int>(buf), readBUF<int>(buf), readBUF<int>(buf),
                                                    readBUF<int>(buf), readBUF<int>(buf),
                                                    readBUF<int>(buf),readBUF<int>(buf),readBUF<int>(buf),readBUF<int>(buf),
                                                    readBUF<int>(buf),readBUF<int>(buf),readBUF<int>(buf),readBUF<int>(buf),
                                                    nullptr); 
        dnnType *aus = new dnnType[r->chunk_dim*2];
        for(int i=0; i<r->chunk_dim*2; i++)
    		aus[i] = readBUF<dnnType>(buf);
		checkCuda( cudaMemcpy(r->offset, aus, sizeof(dnnType)*2*r->chunk_dim, cudaMemcpyHostToDevice) );
        free(aus);
		aus = new dnnType[r->chunk_dim];
		for(int i=0; i<r->chunk_dim; i++)
            aus[i] = readBUF<dnnType>(buf);
		checkCuda( cudaMemcpy(r->mask, aus, sizeof(dnnType)*r->chunk_dim, cudaMemcpyHostToDevice) );
        free(aus);
		aus = new dnnType[(r->i_c * r->o_c * r->kh * r->kw * 1 )];
		for(int i=0; i<(r->i_c * r->o_c * r->kh * r->kw * 1 ); i++)
    		aus[i] = readBUF<dnnType>(buf);
		checkCuda( cudaMemcpy(r->data_d, aus, sizeof(dnnType)*(r->i_c * r->o_c * r->kh * r->kw * 1 ), cudaMemcpyHostToDevice) );
        free(aus);
		aus = new dnnType[r->o_c];
		for(int i=0; i < r->o_c; i++)
    		aus[i] = readBUF<dnnType>(buf);
		checkCuda( cudaMemcpy(r->bias2_d, aus, sizeof(dnnType)*r->o_c, cudaMemcpyHostToDevice) );
        free(aus);
		aus = new dnnType[r->height_ones * r->width_ones];
		for(int i=0; i<r->height_ones * r->width_ones; i++)
    		aus[i] = readBUF<dnnType>(buf);
		checkCuda( cudaMemcpy(r->ones_d1, aus, sizeof(dnnType)*r->height_ones * r->width_ones, cudaMemcpyHostToDevice) );
        free(aus);
		aus = new dnnType[r->dim_ones];
		for(int i=0; i<r->dim_ones; i++)
    		aus[i] = readBUF<dnnType>(buf);
		checkCuda( cudaMemcpy(r->ones_d2, aus, sizeof(dnnType)*r->dim_ones, cudaMemcpyHostToDevice) );
        free(aus);
        return r;
    } 

    FatalError("Cant deserialize Plugin");
    return NULL;
}

}}

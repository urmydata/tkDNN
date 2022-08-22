#ifndef NETWORKRT_H
#define NETWORKRT_H

#include <string.h> // memcpy
#include <set>
#include "utils.h"
#include "Network.h"
#include "Layer.h"
#include "NvInfer.h"
#include <memory>
#include <tkDNN/kernels.h>
#include <pluginsRT/ActivationLeakyRT.h>
#include <pluginsRT/ActivationLogisticRT.h>
#include <pluginsRT/ActivationMishRT.h>
#include <pluginsRT/ActivationReLUCeilingRT.h>
#include <pluginsRT/DeformableConvRT.h>
#include <pluginsRT/FlattenConcatRT.h>
#include <pluginsRT/MaxPoolingFixedSizeRT.h>
#include <pluginsRT/RegionRT.h>
#include <pluginsRT/ReorgRT.h>
#include <pluginsRT/ReshapeRT.h>
#include <pluginsRT/ResizeLayerRT.h>
#include <pluginsRT/RouteRT.h>
#include <pluginsRT/ShortcutRT.h>
#include <pluginsRT/UpsampleRT.h>
#include <pluginsRT/YoloRT.h>



namespace tk { namespace dnn {

using namespace nvinfer1;



class NetworkRT {

public:
    nvinfer1::DataType dtRT;
    nvinfer1::IBuilder *builderRT;
    nvinfer1::IRuntime *runtimeRT;
    nvinfer1::INetworkDefinition *networkRT; 
#if NV_TENSORRT_MAJOR >= 6  
    nvinfer1::IBuilderConfig *configRT;
#endif
    
    nvinfer1::ICudaEngine *engineRT;
    nvinfer1::IExecutionContext *contextRT;

    const static int MAX_BUFFERS_RT = 10;
    void* buffersRT[MAX_BUFFERS_RT];
    dataDim_t buffersDIM[MAX_BUFFERS_RT];
    int buf_input_idx, buf_output_idx;
    bool builderActive = false;
	bool is_dla = false;
	bool is_int8 = false;
	int startIndex = 0;
    dataDim_t input_dim, output_dim;
    dnnType *output;
    cudaStream_t stream;

	NetworkRT(Network *net, const char *name, int start_index, int end_index, int dla_core);
	NetworkRT(Network *net, const char *name);
	static void makeOutputMap(Network *net, std::map<int, std::list<int>>& output_map);
	static std::set<int> getInputLayers(Network *net, int start_index, int end_index);
	static std::map<std::pair<int, int>, int> getInputPair(Network *net, int start_index, int end_index);
	static std::map<std::pair<int, int>, int> getOutputPair(Network *net, int start_index, int end_index);
	virtual ~NetworkRT();

    int getMaxBatchSize() {
        if(engineRT != nullptr)
            return engineRT->getMaxBatchSize();
        else
            return 0;
    }

    int getBuffersN() {
        if(engineRT != nullptr)
            return engineRT->getNbBindings();
        else 
            return 0;
    }

    /**
        Do inference
    */
    dnnType* infer(dataDim_t &dim, dnnType* data);
    void enqueue(int batchSize = 1);    

    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Layer *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Conv2d *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Activation *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Dense *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Pooling *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Softmax *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Route *l);
    nvinfer1::IPluginV2Layer* convert_layer(nvinfer1::ITensor *input, Flatten *l);
    nvinfer1::IPluginV2Layer* convert_layer(nvinfer1::ITensor *input, Reshape *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Resize *l);
    nvinfer1::IPluginV2Layer* convert_layer(nvinfer1::ITensor *input, Reorg *l);
    nvinfer1::IPluginV2Layer* convert_layer(nvinfer1::ITensor *input, Region *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Shortcut *l);
    nvinfer1::IPluginV2Layer* convert_layer(nvinfer1::ITensor *input, Yolo *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Upsample *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, DeformConv2d *l);

#if NV_TENSORRT_MAJOR > 5 && NV_TENSORRT_MAJOR < 8
    bool serialize(const char *filename);
#else
    bool serialize(const char *filename,nvinfer1::IHostMemory *ptr);
#endif

    bool deserialize(const char *filename);
    bool deserialize(const char *filename, int dla_core);
    void destroy();

	void run_on_dla(ILayer*l);
};


}}
#endif //NETWORKRT_H

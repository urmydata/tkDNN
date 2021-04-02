#include<cassert>
#include "../kernels.h"

#include <cublas_v2.h>

class ShortcutRT : public IPluginExt {

public:
	ShortcutRT(tk::dnn::dataDim_t bdim) {
		this->bc = bdim.c;
		this->bh = bdim.h;
		this->bw = bdim.w;
	}

	~ShortcutRT(){

	}

	int getNbOutputs() const override {
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override {
		return DimsCHW{inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]};
	}

	/*void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override {
		c = inputDims[0].d[0];
		h = inputDims[0].d[1];
		w = inputDims[0].d[2];
	}*/

	int initialize() override {

		return 0;
	}

	virtual void terminate() override {
	}

	virtual size_t getWorkspaceSize(int maxBatchSize) const override {
		return 0;
	}

	virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override {
		if(mDataType == nvinfer1::DataType::kFLOAT)
		{
			dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
			dnnType *srcDataBack = (dnnType*)reinterpret_cast<const dnnType*>(inputs[1]);
			dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);

			checkCuda( cudaMemcpyAsync(dstData, srcData, batchSize*c*h*w*sizeof(dnnType), cudaMemcpyDeviceToDevice, stream));
			for(int b=0; b < batchSize; ++b)
				shortcutForward(srcDataBack + b*bc*bh*bw, dstData + b*c*h*w, 1, c, h, w, 1, 1, bc, bh, bw, 1, stream);
		}
		else
		{
			__half *srcData = (__half*)reinterpret_cast<const __half*>(inputs[0]);
			__half *srcDataBack = (__half*)reinterpret_cast<const __half*>(inputs[1]);
			__half *dstData = reinterpret_cast<__half*>(outputs[0]);

			checkCuda( cudaMemcpyAsync(dstData, srcData, batchSize*c*h*w*sizeof(__half), cudaMemcpyDeviceToDevice, stream));
			for(int b=0; b < batchSize; ++b)
				shortcutForwardHalf(srcDataBack + b*bc*bh*bw, dstData + b*c*h*w, 1, c, h, w, 1, 1, bc, bh, bw, 1, stream);
		}

		return 0;
	}


	virtual size_t getSerializationSize() override {
		return 6*sizeof(int) + sizeof(mDataType);
	}

	virtual void serialize(void* buffer) override {
		char *buf = reinterpret_cast<char*>(buffer);
		tk::dnn::writeBUF(buf, bc);
		tk::dnn::writeBUF(buf, bh);
		tk::dnn::writeBUF(buf, bw);
		tk::dnn::writeBUF(buf, c);
		tk::dnn::writeBUF(buf, h);
		tk::dnn::writeBUF(buf, w);
		tk::dnn::writeBUF(buf, mDataType);
	}

	bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const override {
		return ((type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kHALF) && format == nvinfer1::PluginFormat::kNCHW);
	}

	void configureWithFormat(const Dims* inputDims, int32_t nbInputs, const Dims* outputDims, int32_t nbOutputs,
							         nvinfer1::DataType type, nvinfer1::PluginFormat format, int32_t maxBatchSize) override {
		c = inputDims[0].d[0];
		h = inputDims[0].d[1];
		w = inputDims[0].d[2];
		mDataType = type;
	}

	int c, h, w;
	int bc, bh, bw;
	nvinfer1::DataType mDataType{nvinfer1::DataType::kFLOAT};
};

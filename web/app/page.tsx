'use client';

import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/components/ui/select';
import { ImageIcon } from 'lucide-react';
import { calculateEquation, CalculatorResult } from '@/actions/calculate_action';
import Image from 'next/image';
import { BlockMath } from 'react-katex';

export default function Home() {
    const Image_models = [
        { label: 'CNN', value: 'cnn' },
        { label: 'ResNet', value: 'resnet' },
        { label: 'Visual Transformer', value: 'vision_transformer' },
    ];

    const Sequence_models = [
        { label: 'BILSTM', value: 'bilstm' },
        { label: 'GRU', value: 'gru' },
        { label: 'Transformer', value: 'transformer' },
    ];

    const [loading, setLoading] = useState(false);
    const [file, setFile] = useState<File | null>(null);
    const [result, setResult] = useState<CalculatorResult | null>(null);
    const [error, setError] = useState('');

    const [imageModel, setImageModel] = useState(Image_models[0].value);
    const [sequenceModel, setSequenceModel] = useState(Sequence_models[0].value);

    const [previewUrl, setPreviewUrl] = useState<string | null>(null);

    async function handleCalculate() {
        if (!file) {
            setError('Please select an image.');
            return;
        }

        setLoading(true);
        setError('');

        try {
            const formData = new FormData();

            formData.append('image', file);
            formData.append('imageModel', imageModel);
            formData.append('sequenceModel', sequenceModel);

            const response = await calculateEquation(formData);

            if (response.status === 'error') {
                setError(response.error ?? 'Something went wrong.');
                setResult(null);
            } else {
                setResult(response);
            }
        } catch (error) {
            console.error(error);

            setError('Something went wrong while processing the image.');
            setResult(null);
        } finally {
            setLoading(false);
        }
    }

    useEffect(() => {
        return () => {
            if (previewUrl) {
                URL.revokeObjectURL(previewUrl);
            }
        };
    }, [previewUrl]);

    return (
        <div className=" max-w-7xl flex flex-col gap-8 py-4 px-8 w-full mx-auto">
            <div>
                <h1 className="text-black font-bold text-3xl">
                    Recognize and solve{' '}
                    <span className="text-[#0051D5] font-bold text-3xl">handwritten equations</span>
                </h1>
                <p className="text-[#45464D] text-base">
                    Upload an equation image, choose a model combination, and calculate the result.
                </p>
            </div>

            <div className="flex gap-6">
                <div className="flex flex-col gap-5 bg-white p-5 flex-1">
                    <h2 className="text-[#002B75] font-semibold text-lg">Equation input</h2>
                    <hr className="border-t border-[#C6C6CD]" />
                    <div
                        className={`relative flex flex-1 min-h-48 flex-col items-center justify-center gap-2 rounded-md border-2 border-dashed border-[#C6C6CD] bg-[#EFF4FF] p-6 text-center
                            ${loading ? 'opacity-50' : ''}`}
                    >
                        {previewUrl ? (
                            <>
                                <Image
                                    src={previewUrl}
                                    alt="Equation preview"
                                    width={500}
                                    height={200}
                                    unoptimized
                                    className="max-h-40 w-auto object-contain rounded-md"
                                />

                                <p className="text-sm font-medium text-[#0B1C30]">{file?.name}</p>
                            </>
                        ) : (
                            <>
                                <ImageIcon className="size-8 text-[#0051D5]" aria-hidden="true" />

                                <p className="text-lg font-medium text-[#0B1C30]">
                                    Choose an image or browse
                                </p>

                                <p className="text-sm font-normal text-[#45464D]">PNG or JPG</p>
                            </>
                        )}
                        <Input
                            className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
                            type="file"
                            accept="image/png,image/jpeg"
                            aria-label="Upload an equation image"
                            disabled={loading}
                            onChange={(event) => {
                                const selectedFile = event.target.files?.[0];

                                setError('');
                                setResult(null);
                                setFile(null);
                                setPreviewUrl(null);

                                if (!selectedFile) return;

                                if (!['image/png', 'image/jpeg'].includes(selectedFile.type)) {
                                    setError('Please select a PNG or JPEG image.');
                                    event.target.value = '';
                                    return;
                                }

                                if (selectedFile.size > 4 * 1024 * 1024) {
                                    setError('Please select an image smaller than 4 MB.');
                                    event.target.value = '';
                                    return;
                                }

                                setFile(selectedFile);
                                setPreviewUrl(URL.createObjectURL(selectedFile));
                            }}
                        />
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                        <div className="flex flex-col gap-1">
                            <h3 className="text-[#002B75] font-semibold text-xs">
                                Image analysis model
                            </h3>
                            <p className="text-xs text-[#45464D]">
                                Finds visual patterns in the handwriting.
                            </p>
                            <Select
                                items={Image_models}
                                value={imageModel}
                                onValueChange={(value) => {
                                    if (value !== null) {
                                        setImageModel(value);
                                    }
                                }}
                            >
                                <SelectTrigger className="w-full">
                                    <SelectValue placeholder="Select a model" />
                                </SelectTrigger>
                                <SelectContent>
                                    {Image_models.map((model) => (
                                        <SelectItem
                                            key={model.value}
                                            value={model.value}
                                            className="pl-2.75"
                                        >
                                            {model.label}
                                        </SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                        </div>
                        <div className="flex flex-col gap-1">
                            <h3 className="text-[#002B75] font-semibold text-xs">
                                Sequence reading model
                            </h3>
                            <p className="text-xs text-[#45464D]">
                                Reads the detected features in order.
                            </p>
                            <Select
                                items={Sequence_models}
                                value={sequenceModel}
                                onValueChange={(value) => {
                                    if (value !== null) {
                                        setSequenceModel(value);
                                    }
                                }}
                            >
                                <SelectTrigger className="w-full">
                                    <SelectValue placeholder="Select a model" />
                                </SelectTrigger>
                                <SelectContent>
                                    {Sequence_models.map((model) => (
                                        <SelectItem
                                            key={model.value}
                                            value={model.value}
                                            className="pl-2.75"
                                        >
                                            {model.label}
                                        </SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                        </div>
                    </div>

                    <Button
                        className="bg-[#0051D5] text-white py-2.5 rounded-md"
                        onClick={handleCalculate}
                        disabled={loading}
                    >
                        {loading ? 'Calculating...' : 'Calculate'}
                    </Button>
                </div>
                <div className="flex flex-col gap-5 bg-white p-5 max-w-125">
                    <h2 className="text-[#002B75] font-semibold text-lg">How it works</h2>
                    <hr className="border-t border-[#C6C6CD]" />
                    <div className="flex flex-col gap-3">
                        <div className="bg-[#EFF4FF] flex flex-col gap-1 p-3 rounded-md">
                            <h2 className="text-lg font-semibold text-[#0051D5]">Upload</h2>
                            <p className="text-xs text-[#45464D]">
                                Add a clear image of one handwritten expression.
                            </p>
                        </div>
                        <div className="bg-[#EFF4FF] flex flex-col gap-1 p-3 rounded-md">
                            <h2 className="text-lg font-semibold text-[#0051D5]">Configure</h2>
                            <p className="text-xs text-[#45464D]">
                                Select an image and sequence model combination.
                            </p>
                        </div>
                        <div className="bg-[#EFF4FF] flex flex-col gap-1 p-3 rounded-md">
                            <h2 className="text-lg font-semibold text-[#0051D5]">Calculate</h2>
                            <p className="text-xs text-[#45464D]">
                                Review the recognized expression and final answer.
                            </p>
                        </div>
                    </div>
                    <hr className="border-t border-[#C6C6CD]" />
                    <div className="flex flex-col gap-1">
                        <h2 className="text-lg font-semibold text-[#002B75]">Supported notation</h2>
                        <p className="text-xs">
                            Digits 0–9, addition (+), subtraction (−), multiplication (×), division
                            (÷), and parentheses ( ).
                        </p>
                        <p className="text-xs text-[#45464D]">
                            Other symbols and mathematical notation are not supported.
                        </p>
                    </div>
                </div>
            </div>
            <div className="flex flex-col gap-5 bg-white p-5">
                <h2 className="text-lg font-semibold text-[#002B75]">Result</h2>

                <hr className="border-t border-[#C6C6CD]" />
                {!result && !error && (
                    <p className="text-sm text-[#45464D]">
                        Your recognized expression and calculated answer will appear here.
                    </p>
                )}
                {error && <p className="text-sm text-red-600">{error}</p>}

                {result && result.status === 'success' && (
                    <div className="flex w-full gap-3 bg-[#EFF4FF] rounded-md">
                        <div className="flex flex-col gap-1 p-3 flex-1">
                            <p className="text-xs text-[#0051D5]">Recognized expression</p>

                            <div className="text-lg text-[#0B1C30]">
                                <BlockMath math={result.equation ?? ''} />
                            </div>
                        </div>

                        <div className="flex flex-col gap-1 p-3 max-w-50">
                            <p className="text-xs text-[#0051D5]">Calculated answer</p>

                            <h3 className="text-lg text-[#0B1C30]">{result.answer}</h3>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

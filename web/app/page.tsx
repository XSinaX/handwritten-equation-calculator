'use client';

import { useState } from 'react';
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

export default function Home() {
    //TODO: get the model options from the backend
    const Image_models = [
        { label: 'Basic CNN', value: 'basic_cnn' },
        { label: 'ResNet', value: 'resnet' },
        { label: 'Visual Transformer', value: 'visual_transformer' },
    ];
    const Sequence_models = [
        { label: 'LSTM', value: 'lstm' },
        { label: 'GRU', value: 'gru' },
        { label: 'Transformer', value: 'transformer' },
    ];
    const [loading, setLoading] = useState(false);
    const [file, setFile] = useState<File | null>(null);
    const [result, setResult] = useState<unknown>(null);
    const [error, setError] = useState('');

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
                        <ImageIcon className="size-8 text-[#0051D5]" aria-hidden="true" />

                        <p className="text-lg font-medium text-[#0B1C30]">
                            {file ? file.name : 'Choose an image or browse'}
                        </p>

                        <p className="text-sm font-normal text-[#45464D]">PNG or JPG</p>
                        {/*TODO: complete the file upload functionality */}
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

                                if (!selectedFile) return;

                                if (!['image/png', 'image/jpeg'].includes(selectedFile.type)) {
                                    setError('Please select a PNG or JPEG image.');
                                    event.target.value = '';
                                    return;
                                }

                                setFile(selectedFile);
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
                            <Select items={Image_models} defaultValue={Image_models[0].value}>
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
                            <Select items={Sequence_models} defaultValue={Sequence_models[0].value}>
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

                    <Button className="bg-[#0051D5] text-white py-2.5 rounded-md">
                        Recognize and calculate
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
                <div>
                    <h2 className="text-lg font-semibold text-[#002B75]">Result</h2>
                    <p className="text-xs text-[#45464D]">
                        Your recognized expression and calculated answer.
                    </p>
                </div>
                <hr className="border-t border-[#C6C6CD]" />
                <div className="flex w-full gap-3 bg-[#EFF4FF] rounded-md">
                    <div className="flex flex-col gap-1  p-3 flex-1">
                        <p className="text-xs text-[#0051D5]">Recognized expression</p>
                        <h3 className="text-lg text-[#0B1C30]">2 + 2 × 3</h3>{' '}
                        {/*TODO: replace with actual recognized expression */}
                    </div>
                    <div className="flex flex-col gap-1  p-3 max-w[200px]">
                        <p className="text-xs text-[#0051D5]">Calculated answer</p>
                        <h3 className="text-lg text-[#0B1C30]">8</h3>{' '}
                        {/*TODO: replace with actual answer */}
                    </div>
                </div>
            </div>
        </div>
    );
}

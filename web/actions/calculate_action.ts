'use server';

export type CalculatorResult = {
    status: 'success' | 'error';
    equation?: string;
    answer?: number;
    error?: string;
};

export async function calculateEquation(formData: FormData): Promise<CalculatorResult> {
    const image = formData.get('image');
    const architecture = formData.get('architecture');
    const model = formData.get('model');

    if (!(image instanceof File)) {
        return {
            status: 'error',
            error: 'No image was provided.',
        };
    }

    if (architecture !== 'end_to_end' && architecture !== 'segmentation') {
        return {
            status: 'error',
            error: 'Invalid recognition architecture.',
        };
    }

    if (typeof model !== 'string' || !model.trim()) {
        return {
            status: 'error',
            error: 'Please select a model.',
        };
    }

    const backendFormData = new FormData();

    backendFormData.append('image', image);
    backendFormData.append('architecture', architecture);
    backendFormData.append('model', model);

    try {
        const response = await fetch(`${process.env.BACKEND_URL}/calculate`, {
            method: 'POST',
            body: backendFormData,
            cache: 'no-store',
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => null);

            return {
                status: 'error',
                error:
                    typeof errorData?.detail === 'string'
                        ? errorData.detail
                        : 'Failed to process the image.',
            };
        }

        const data = await response.json();

        return {
            status: 'success',
            equation: data.equation,
            answer: data.answer,
        };
    } catch (error) {
        console.error(error);

        return {
            status: 'error',
            error: 'Could not connect to the backend.',
        };
    }
}

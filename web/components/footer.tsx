import Link from 'next/link';

export default function Footer() {
    return (
        <footer className="w-full bg-[#EFF4FF] border-t border-[#C6C6CD] ">
            <div className="flex justify-between items-center p-8 mx-auto max-w-7xl">
                <p className="text-xs text-[#45464D]">
                    Modular equation recognition research project.
                </p>
                <div className="flex gap-4 items-center justify-center">
                    <Link
                        className="text-xs text-45464D]"
                        href="https://github.com/XSinaX/handwritten-equation-calculator"
                    >
                        Github
                    </Link>
                    <Link
                        className="text-xs text-45464D]"
                        href="https://github.com/XSinaX/handwritten-equation-calculator/blob/main/README.md"
                    >
                        Documentation
                    </Link>
                </div>
            </div>
        </footer>
    );
}

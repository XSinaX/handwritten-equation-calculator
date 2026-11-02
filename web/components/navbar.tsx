import Link from 'next/link';
import Image from 'next/image';

export default function Navbar() {
    return (
        <nav className="w-full bg-[#F8F9FF]/90 border-b border-[#C6C6CD] backdrop-blur-xl">
            <div className="flex justify-between max-w-7xl px-8 py-5 mx-auto">
                <div className="flex gap-2 items-center">
                    <h1 className="text-lg font-semibold text-[#0B1C30]">InkSolve</h1>
                    <Image src="/logo.svg" alt="InkSolve Logo" width={32} height={32} />
                </div>
                <div>
                    <Link
                        href="/"
                        className="inline-flex items-center justify-center rounded-md px-4 py-2 text-sm font-semibold hover:bg-[#DCE9FF] hover:text-[#0051D5]"
                    >
                        Calculator
                    </Link>
                </div>
            </div>
        </nav>
    );
}
